import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter,rotate,map_coordinates,filters
from cross_correlation_letter_detector.improccissing_utilities import correlate, normalise_kernel_in_place, calculate_normalised_crosscorrelation
import itertools
import pickle
from cross_correlation_letter_detector.grid_iterator import grid_iterator_from_corners
import tqdm

b_load_last_run = True


def do_ccr_v1(path, letters_db, path_for_dump_files, frame_char, corner_char):
    try:
        assert b_load_last_run
        _, grid_letter_match_results, letters, = np.load(path_for_dump_files + ".npz").values()
    except:
        letters, grid_letter_match_results = get_grid_letter_match_results(path, letters_db, frame_char, corner_char)


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

    # find corners
    # c0, c1, c2, c3 = find_corners(document_image, corner_char, frame_char, letters_db)
    c0, c1, c2, c3 = (np.array([230, 186]),
                      np.array([4848,  137]),
                      np.array([ 273, 6720]),
                      np.array([4892, 6679]))
    # rotate
    p1 = (c0 + c1) / 2
    p2 = (c2 + c3) / 2
    rotation_angle = np.arctan2(*(p1 - p2)) % np.pi

    def rotate_point(p,angle,center):
        return np.matmul([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], p - center) + center

    document_image = rotate(document_image, -rotation_angle * 180 / np.pi,cval=np.max(document_image))
    old_document_center = np.array(document_image.shape) / 2
    c0 = rotate_point(c0,rotation_angle,old_document_center)
    c1 = rotate_point(c1,rotation_angle,old_document_center)
    c2 = rotate_point(c2,rotation_angle,old_document_center)
    c3 = rotate_point(c3,rotation_angle,old_document_center)

    # find frame

    frame_char_image = letters_db[frame_char]
    frame_char_fit = calculate_normalised_crosscorrelation(document_image, frame_char_image)
    frame_char_fit_g = gaussian_filter(frame_char_fit ** 4, 5)

    def get_two_largest_local_maxima(profile):
        temp = abs(profile[1:-1] * np.diff(np.sign(np.diff(profile))))
        idx1 = np.argmax(temp)
        temp[idx1] = 0
        idx2 = np.argmax(temp)
        return min(idx1, idx2) + 1, max(idx1, idx2) + 1

    x_profile = np.mean(frame_char_fit_g, axis=0)
    frameL, frameR = get_two_largest_local_maxima(x_profile)

    y_profile = np.mean(frame_char_fit_g, axis=1)
    frameU, frameD = get_two_largest_local_maxima(y_profile)

    # document_image_full = document_image
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
        letter_image = letters_db[letter_char]
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

def find_corners(document_image,corner_char,frame_char,letters_db):
    document_size = np.sum(np.array(document_image.shape) ** 2) ** 0.5
    document_area = np.product(document_image.shape)

    def fuzzy_sign(x, a=0.01):
        return x / (np.abs(x) + a)

    def score(corner_char_fit, shifted_frame_char_fit):
        return (fuzzy_sign(corner_char_fit - shifted_frame_char_fit) + 1) * corner_char_fit

    corner_char_image = letters_db[corner_char]
    corner_char_fit = calculate_normalised_crosscorrelation(document_image, corner_char_image)

    frame_char_image = letters_db[frame_char]
    shifted_frame_char_image = np.roll(frame_char_image, frame_char_image.shape[1] // 2, axis=1)
    shifted_frame_char_fit = calculate_normalised_crosscorrelation(document_image, shifted_frame_char_image)

    corner_scores = gaussian_filter(score(corner_char_fit, shifted_frame_char_fit) ** 3, 5)

    y, x = np.where((filters.maximum_filter(corner_scores, document_size // 20) == corner_scores) * (corner_scores > 0.1)) # todo 20 to 80

    indexes_up = np.full(x.shape, True)
    indexes_down = np.full(x.shape, True)
    indexes_left = np.full(x.shape, True)
    indexes_right = np.full(x.shape, True)

    indexes_up[y > document_image.shape[0] / 3] = False
    indexes_down[y < document_image.shape[0] * 2 / 3] = False
    indexes_left[x > document_image.shape[1] / 3] = False
    indexes_right[x < document_image.shape[1] * 2 / 3] = False

    plt.scatter(x, y)

    points_group_0 = list(zip(x[indexes_up * indexes_left], y[indexes_up * indexes_left]))
    points_group_1 = list(zip(x[indexes_up * indexes_right], y[indexes_up * indexes_right]))
    points_group_2 = list(zip(x[indexes_down * indexes_left], y[indexes_down * indexes_left]))
    points_group_3 = list(zip(x[indexes_down * indexes_right], y[indexes_down * indexes_right]))

    def get_distance(a, b):
        return np.sum((a - b) ** 2) ** 0.5

    def score_corners(c0, c1, c2, c3):
        c0, c1, c2, c3 = np.array(c0), np.array(c1), np.array(c2), np.array(c3)

        # is rectangle
        is_rectangle_score = abs(get_distance(c0, c1) - get_distance(c2, c3)) + \
                             abs(get_distance(c0, c2) - get_distance(c1, c3)) + \
                             abs(get_distance(c0, c3) - get_distance(c1, c2))
        is_rectangle_score = is_rectangle_score < document_size / 20

        # rectangle size
        rectangle_area = abs(np.cross((c1 - c0), (c2 - c0))) / (2 * document_area) + \
                         abs(np.cross((c1 - c3), (c2 - c3))) / (2 * document_area)

        # values
        char_score = min(corner_scores[[c0[1], c1[1], c2[1], c3[1]], [c0[0], c1[0], c2[0], c3[0]]])

        return is_rectangle_score, char_score * rectangle_area

    best_corners_score = (False, -float("inf"))
    best_corners = None
    for p0 in tqdm.tqdm(points_group_0):
        for p1 in points_group_1:
            for p2 in points_group_2:
                for p3 in points_group_3:
                    cur_score = score_corners(p0, p1, p2, p3)
                    if cur_score > best_corners_score:
                        best_corners_score = cur_score
                        best_corners = np.array([p0, p1, p2, p3])

    return best_corners

if __name__ == "__main__":
    do_ccr_v1(r"E:\Liors_Stuff\--- Long Term Storage\Open University\לימודי טעודה\2021\מעבדה בתכנות מערכות 20465\hw1\scan\p1.jpg",
              r"cross_correlation_letter_detector\Consolas8_600dpi_db.pickle",None,frame_char="X",corner_char="O")

