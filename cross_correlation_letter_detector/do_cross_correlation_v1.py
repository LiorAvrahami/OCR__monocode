import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter,rotate,map_coordinates
from .improccissing_utilities import correlate, normalise_kernel_in_place, calculate_normalised_crosscorrelation
import itertools
import pickle
from .grid_iterator import grid_iterator_from_corners
import tqdm

b_load_last_run = False


def do_ccr_v1(path, letters_db, path_for_dump_files, frame_char, corner_char):
    if not b_load_last_run:
        letters,grid_letter_match_results = get_grid_letter_match_results(path, letters_db, frame_char, corner_char)
    else:
        _, grid_letter_match_results, letters,  = np.load(path_for_dump_files + ".npz").values()

    chosen_letters_indexes_arr = np.argmax(grid_letter_match_results, axis=-1)

    # fix spaces taking over
    space_char_index = np.where(letters == " ")[0]
    chosen_letters_indexes_arr_without_space = np.argmax(grid_letter_match_results[:, :, letters != " "], axis=-1)
    x,y = np.meshgrid(range(chosen_letters_indexes_arr_without_space.shape[0]),range(chosen_letters_indexes_arr_without_space.shape[1]),indexing="ij")
    chosen_grid_match_results_without_spaces =  grid_letter_match_results[x.flatten(),y.flatten(),chosen_letters_indexes_arr_without_space.flatten()].reshape(x.shape)
    false_space_indexes = (chosen_letters_indexes_arr == space_char_index) * (chosen_grid_match_results_without_spaces > 0.7)
    chosen_letters_indexes_arr[false_space_indexes] = chosen_letters_indexes_arr_without_space[false_space_indexes]

    average_score = np.mean(grid_letter_match_results[chosen_letters_indexes_arr])
    result_text = letters[chosen_letters_indexes_arr].T
    result_text = "\n".join(["".join(ln) for ln in result_text])

    if path_for_dump_files is not None:
        np.savez(path_for_dump_files, result_text, grid_letter_match_results, letters)

    return average_score, result_text

def get_grid_letter_match_results(path,letters_db,frame_char,corner_char):
    b_rotate = False
    partial_cells_overlap = -0.5  # should be between 0 and 3, make it big if text is not very grid-like,
    b_plot_grid = False

    document_image = plt.imread(path)[:, :, 0].astype(float)

    frame_char_image = np.mean(letters_db[frame_char][10:20], axis=0)
    frame_char_fit = calculate_normalised_crosscorrelation(document_image, frame_char_image)
    frame_char_fit_g = gaussian_filter(frame_char_fit ** 4, 5)

    def get_two_largest_local_maxima(profile):
        temp = abs(profile[1:-1] * np.diff(np.sign(np.diff(profile))))
        idx1 = np.argmax(temp)
        temp[idx1] = 0
        idx2 = np.argmax(temp)
        return min(idx1, idx2), max(idx1, idx2)

    # rotate TODO

    # find frame
    x_profile = np.mean(frame_char_fit_g, axis=0)
    frameL, frameR = get_two_largest_local_maxima(x_profile)

    y_profile = np.mean(frame_char_fit_g, axis=1)
    frameU, frameD = get_two_largest_local_maxima(y_profile)

    document_image = document_image[int(frameU):int(np.ceil(frameD)), int(frameL):int(np.ceil(frameR))]

    frame_vertices = (frameL % 1,frameU % 1), (frameR - int(frameL),frameU % 1), (frameL % 1,frameD - int(frameU)), \
                     (frameR - int(frameL), frameD - int(frameU))

    axes_to_plot = None
    if b_plot_grid:
        axes_to_plot = plt.axes()
        axes_to_plot.imshow(document_image)
        axes_to_plot.set_title("examine")

    # find num of rows, columnes TODO
    num_of_rows = 77
    num_of_columns = 125

    letters = np.array(list(letters_db.keys()))
    grid_letter_match_results = np.full((num_of_columns, num_of_rows, len(letters),), np.nan, np.float)
    for letter_index, letter_char in enumerate(tqdm.tqdm(letters, position=0)):
        letter_image = letters_db[letter_char][0]
        char_fit_arr = calculate_normalised_crosscorrelation(document_image, letter_image)

        grid_iterator = grid_iterator_from_corners(
            num_of_rows=num_of_rows + 2, num_of_columns=num_of_columns + 2, quadrilateral_vertices=frame_vertices,
            partial_cells_overlap=partial_cells_overlap,
            b_plot=b_plot_grid, axes_object_to_plot=axes_to_plot)

        for (x_idx, y_idx), box in grid_iterator:
            if x_idx == 0 or y_idx == 0 or x_idx == num_of_columns + 1 or y_idx == num_of_rows + 1:
                continue
            # if y_idx == 1 or y_idx > 2 or x_idx > 20:#todo remove
            #     continue

            grid_cell_char_fit_arr = char_fit_arr[int(box[1][0]):int(np.ceil(box[1][1])),
                                     int(box[0][0]):int(np.ceil(box[0][1]))]
            grid_letter_match_results[x_idx - 1, y_idx - 1, letter_index] = np.max(grid_cell_char_fit_arr)
    return letters,grid_letter_match_results


if __name__ == "__main__":
    do_ccr_v1(r"E:\Liors_Stuff\--- Long Term Storage\Open University\לימודי טעודה\2021\מעבדה בתכנות מערכות 20465\hw1\scan\p1.jpg",
              r"cross_correlation_letter_detector\Consolas8_600dpi_db.pickle",None,frame_char="X",corner_char="O")

