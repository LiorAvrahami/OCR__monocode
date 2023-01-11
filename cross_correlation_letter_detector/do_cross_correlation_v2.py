import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, rotate, map_coordinates, filters
from .improccissing_utilities import correlate, normalise_kernel_in_place, calculate_normalised_crosscorrelation
import itertools
import pickle
from .grid_iterator import grid_iterator_from_corners
import tqdm

from find_corners import find_corners


def do_ccr_v2(path, letters_db, path_for_dump_files, frame_char, corner_char, b_load_last_run, b_skip_ocr=False):
    try:
        assert b_load_last_run
        _, grid_letter_match_results, letters, = np.load(path_for_dump_files + ".npz").values()
    except:
        if not b_skip_ocr:
            letters, grid_letter_match_results = get_grid_letter_match_results(path, letters_db, frame_char, corner_char)

    # ignore nans (TODO why do I sometimes randomly get unrepreducable nan in this array!? is this some statistical electrical fault?)
    grid_letter_match_results_no_nans = np.array(grid_letter_match_results)
    grid_letter_match_results_no_nans[np.isnan(grid_letter_match_results_no_nans)] = 0

    chosen_letters_indexes_arr = np.argmax(grid_letter_match_results_no_nans, axis=-1)

    # fix spaces taking over
    space_char_index = np.where(letters == " ")[0]
    chosen_letters_indexes_arr_without_space = np.argmax(grid_letter_match_results_no_nans[:, :, letters != " "], axis=-1)
    x, y = np.meshgrid(range(chosen_letters_indexes_arr_without_space.shape[0]), range(chosen_letters_indexes_arr_without_space.shape[1]), indexing="ij")
    chosen_grid_match_results_without_spaces = grid_letter_match_results_no_nans[
        x.flatten(), y.flatten(), chosen_letters_indexes_arr_without_space.flatten()].reshape(x.shape)
    false_space_indexes = (chosen_letters_indexes_arr == space_char_index) * (chosen_grid_match_results_without_spaces > 0.7)
    chosen_letters_indexes_arr[false_space_indexes] = chosen_letters_indexes_arr_without_space[false_space_indexes]

    average_score = np.mean(grid_letter_match_results_no_nans[chosen_letters_indexes_arr])
    result_text = letters[chosen_letters_indexes_arr].T
    result_text = "\n".join(["".join(ln) for ln in result_text])

    if path_for_dump_files is not None:
        np.savez(path_for_dump_files, result_text, grid_letter_match_results, letters)

    return average_score, result_text


def get_grid_letter_match_results(path, letters_db, frame_char, corner_char):
    partial_cells_overlap = -0.8  # should be between -1 and 3, make it big if text is not very grid-like,
    b_plot_grid = True

    try:
        document_image = plt.imread(path)[:, :, 0].astype(float)
    except IndexError:
        document_image = plt.imread(path)[:, :].astype(float)

    # find corners
    c0, c1, c2, c3 = find_corners(document_image, corner_char, frame_char, letters_db)

    # rotate
    p1 = (c0 + c1) / 2
    p2 = (c2 + c3) / 2
    rotation_angle = (np.arctan2(*(p1 - p2)) + np.pi / 2) % np.pi - np.pi / 2

    def rotate_point(p, angle, center):
        return np.matmul([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], p - center) + center

    old_document_center = (np.array(document_image.shape) / 2)[[1, 0]]
    document_image = rotate(document_image, -rotation_angle * 180 / np.pi, cval=np.max(document_image))
    new_document_center = (np.array(document_image.shape) / 2)[[1, 0]]
    corner_points = rotate_point(c0, rotation_angle, old_document_center) + new_document_center - old_document_center, \
                    rotate_point(c1, rotation_angle, old_document_center) + new_document_center - old_document_center, \
                    rotate_point(c2, rotation_angle, old_document_center) + new_document_center - old_document_center, \
                    rotate_point(c3, rotation_angle, old_document_center) + new_document_center - old_document_center

    # find frame
    def get_two_largest_local_maxima(profile):
        temp = abs(profile[1:-1] * np.diff(np.sign(np.diff(profile))))
        idx1 = np.argmax(temp)
        temp[idx1] = 0
        idx2 = np.argmax(temp)
        return min(idx1, idx2) + 1, max(idx1, idx2) + 1

    axes_to_plot = None
    if b_plot_grid:
        axes_to_plot = plt.axes()
        axes_to_plot.imshow(document_image)
        axes_to_plot.set_title("examine")

    # find num of rows, columnes TODO
    num_of_rows = 77
    num_of_columns = 125

    letters = np.array(list(letters_db.keys()))
    grid_letter_match_results = np.full((num_of_columns, num_of_rows, len(letters),), np.nan, float)
    for letter_index, letter_char in enumerate(tqdm.tqdm(letters, position=0)):
        letter_image = letters_db[letter_char]
        char_fit_arr = calculate_normalised_crosscorrelation(document_image, letter_image)

        grid_iterator = find_grid_from_frame(num_of_rows=num_of_rows + 2, num_of_columns=num_of_columns + 2, corner_points=corner_points,
                                             document_image=document_image, frame_char=frame_char, letters_db=letters_db,
                                             partial_cells_overlap=partial_cells_overlap, b_plot=b_plot_grid)

        for (x_idx, y_idx), box in grid_iterator:
            if x_idx == 0 or y_idx == 0 or x_idx == num_of_columns + 1 or y_idx == num_of_rows + 1:
                continue

            grid_cell_char_fit_arr = char_fit_arr[int(box[1][0]):int(np.ceil(box[1][1])),
                                     int(box[0][0]):int(np.ceil(box[0][1]))]
            grid_letter_match_results[x_idx - 1, y_idx - 1, letter_index] = np.max(grid_cell_char_fit_arr)
    return letters, grid_letter_match_results


def find_grid_from_frame(num_of_rows, num_of_columns, corner_points, document_image, frame_char, letters_db, partial_cells_overlap: float = 0, b_plot=True):
    frame_char_image = letters_db[frame_char]
    frame_char_fit = calculate_normalised_crosscorrelation(document_image, frame_char_image)
    frame_char_fit_g = gaussian_filter(frame_char_fit ** 3, 5)

    dx = (corner_points[3][0] + corner_points[1][0] - corner_points[0][0] - corner_points[2][0]) / (2 * num_of_columns)
    dy = (corner_points[3][1] + corner_points[2][1] - corner_points[0][1] - corner_points[1][1]) / (2 * num_of_rows)

    def do_single_edge(start_point, end_point, num_of_points):
        grid_points = np.full((num_of_points, 2), np.nan)
        step = (end_point - start_point) / num_of_points
        cur_location = np.array(start_point)
        grid_points[0, :] = start_point
        grid_points[-1, :] = end_point
        for i in range(1, num_of_points - 1):
            cur_location += step
            search_area = frame_char_fit_g[int(cur_location[1] - dy / 2):int(cur_location[1] + dy / 2),
                          int(cur_location[0] - dx / 2):int(cur_location[0] + dx / 2)]
            misalignment = (np.array(next(zip(*np.where(search_area == np.max(search_area))))) - np.array(search_area.shape) / 2)[[1, 0]]
            cur_location += misalignment
            grid_points[i, :] = cur_location
        return grid_points

    left_points = do_single_edge(corner_points[0], corner_points[2], num_of_rows)
    right_points = do_single_edge(corner_points[1], corner_points[3], num_of_rows)
    top_points = do_single_edge(corner_points[0], corner_points[1], num_of_columns)
    bottom_points = do_single_edge(corner_points[2], corner_points[3], num_of_columns)

    if b_plot:
        plt.figure()
        plt.imshow(document_image)
        plt.plot(left_points[:, 0], left_points[:, 1], "ro")
        plt.plot(right_points[:, 0], right_points[:, 1], "ro")
        plt.plot(top_points[:, 0], top_points[:, 1], "ro")
        plt.plot(bottom_points[:, 0], bottom_points[:, 1], "ro")
        for (_, _), box in find_grid_from_frame(num_of_rows, num_of_columns, corner_points, document_image, frame_char, letters_db, partial_cells_overlap,
                                                b_plot=False):
            xb, yb = box
            plt.plot([xb[0], xb[1], xb[1], xb[0], xb[0]], [yb[0], yb[0], yb[1], yb[1], yb[0]], "b")

    # a function with span [0,1] -> [0,1]. wich is curved upwords, made to inflate the input values.
    # the inflation intensity is controlled by the parameter "val_at_half" wich is the functions value at 0.5.
    _inflate = lambda v, p: v / p + 1 - ((1 / p ** 2 - 1) * v ** 2 + 1) ** 0.5
    inflate = lambda v, val_at_half: _inflate(v, 4 * (1 - val_at_half) / (3 - 4 * (1 - val_at_half) ** 2))

    # xfy_weight - left and right's weight in x location average
    proximity_diminance_of_left_right = lambda x, y: (1 - x * (1 - x) * 4) ** 2 / 2 - (1 - y * (1 - y) * 4) ** 2 / 2 + 0.5
    xfy_weight = lambda x, y: inflate(proximity_diminance_of_left_right(x, y), 0.6)
    xfx_weight = lambda x, y: 1 - xfy_weight(x, y)
    yfy_weight = lambda x, y: inflate(proximity_diminance_of_left_right(x, y), 0.95)
    yfx_weight = lambda x, y: 1 - yfy_weight(x, y)

    for y_idx, x_idx in itertools.product(range(num_of_rows), range(num_of_columns)):
        part_x = x_idx / (num_of_columns - 1)
        part_y = y_idx / (num_of_rows - 1)

        # xfx - the x as an average of top and bottom
        xfx = top_points[x_idx][0] * (1 - part_y) + bottom_points[x_idx][0] * (part_y)
        # xfy - the x as an average of left and right
        xfy = left_points[y_idx][0] * (1 - part_x) + right_points[y_idx][0] * (part_x)
        # xfx - the x as an average of top and bottom
        yfx = top_points[x_idx][1] * (1 - part_y) + bottom_points[x_idx][1] * (part_y)
        # xfy - the x as an average of left and right
        yfy = left_points[y_idx][1] * (1 - part_x) + right_points[y_idx][1] * (part_x)

        x = xfx * xfx_weight(part_x, part_y) + xfy * xfy_weight(part_x, part_y)
        y = yfx * yfx_weight(part_x, part_y) + yfy * yfy_weight(part_x, part_y)

        box = (x - dx * (1 + partial_cells_overlap) ** 0.5 / 2, x + dx * (1 + partial_cells_overlap) ** 0.5 / 2), \
            (y - dy * (1 + partial_cells_overlap) ** 0.5 / 2, y + dy * (1 + partial_cells_overlap) ** 0.5 / 2)
        yield (x_idx, y_idx), box
