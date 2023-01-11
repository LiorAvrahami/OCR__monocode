import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, rotate, map_coordinates, filters
from .improccissing_utilities import correlate, normalise_kernel_in_place, calculate_normalised_crosscorrelation
import tqdm


def find_corners(document_image, corner_char, frame_char, letters_db, expected_sides_proportions=1.4142135623730951, min_paper_area_in_image=0.6):
    corner_char_image = letters_db[corner_char]
    frame_char_image = letters_db[frame_char]
    return find_corners_with_char_images(document_image, corner_char_image, frame_char_image, expected_sides_proportions=expected_sides_proportions,
                                         min_paper_area_in_image=min_paper_area_in_image)


def find_corners_with_char_images(document_image, corner_char_image, frame_char_image, expected_sides_proportions=1.4142135623730951,
                                  min_paper_area_in_image=0.6):
    document_size = np.sum(np.array(document_image.shape) ** 2) ** 0.5
    document_area = np.product(document_image.shape)

    def fuzzy_sign(x, a=0.01):
        return x / (np.abs(x) + a)

    def score(corner_char_fit, shifted_frame_char_fit):
        return (fuzzy_sign(corner_char_fit - shifted_frame_char_fit) + 1) * corner_char_fit

    corner_char_fit = calculate_normalised_crosscorrelation(document_image, corner_char_image)

    shifted_frame_char_image = np.roll(frame_char_image, frame_char_image.shape[1] // 2, axis=1)
    shifted_frame_char_fit = calculate_normalised_crosscorrelation(document_image, shifted_frame_char_image)

    corner_scores = gaussian_filter(score(corner_char_fit, shifted_frame_char_fit) ** 3, 5)

    y, x = np.where((filters.maximum_filter(corner_scores, document_size // 60) == corner_scores) * (corner_scores > 0.1))

    indexes_up = np.full(x.shape, True)
    indexes_down = np.full(x.shape, True)
    indexes_left = np.full(x.shape, True)
    indexes_right = np.full(x.shape, True)

    indexes_up[y > document_image.shape[0] / 3] = False
    indexes_down[y < document_image.shape[0] * 2 / 3] = False
    indexes_left[x > document_image.shape[1] / 3] = False
    indexes_right[x < document_image.shape[1] * 2 / 3] = False

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
        is_rectangle_score = np.exp(- 3 * is_rectangle_score / (document_size / 100))

        # rectangle size
        rectangle_area = abs(np.cross((c1 - c0), (c2 - c0))) / (2 * document_area) + \
                         abs(np.cross((c1 - c3), (c2 - c3))) / (2 * document_area)

        # rectangle praportions
        rectangle_praportions_correctness = np.exp(
            - 3 * abs(((get_distance(c0, c2) + get_distance(c1, c3))) / (get_distance(c0, c1) + get_distance(c2, c3)) - expected_sides_proportions) / 0.1)

        # values
        char_score = min(corner_scores[[c0[1], c1[1], c2[1], c3[1]], [c0[0], c1[0], c2[0], c3[0]]])

        score = is_rectangle_score * rectangle_praportions_correctness * (rectangle_area - min_paper_area_in_image + char_score + 0.3) * np.sign(
            rectangle_area - min_paper_area_in_image)

        return score, is_rectangle_score, rectangle_praportions_correctness, char_score, rectangle_area

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

    print(f"rectangle_area = {best_corners_score[-1]:.3g} (expected more then {min_paper_area_in_image:.3g})\n")
    print("rectangle_coordinates:\n{}\n".format(str(best_corners).replace("array", "\narray")))
    return best_corners
