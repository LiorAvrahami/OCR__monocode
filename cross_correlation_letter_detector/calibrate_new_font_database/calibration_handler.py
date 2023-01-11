import efipy
import matplotlib.pyplot as plt
import numpy as np
import safer_prompt_toolkit
from scipy.ndimage import gaussian_filter, rotate, map_coordinates
from scipy.signal import convolve2d  # TODO take from improccissing_utilities
import itertools
import pickle
from cross_correlation_letter_detector.grid_iterator import grid_iterator_from_corners
import tqdm

from ..find_corners import find_corners_with_char_images

from format_files_to_be_printed.code_to_print import make_file_header, put_page_in_frame

# static constants
b_plot = True
VERBOSITY = 2  # determines how much output will be printed


def add_new_font_with_calibration_file(scanned_calibration_file_path):
    num_of_rows = 77 + 2
    num_of_columns = 125 + 2
    m = plt.imread(scanned_calibration_file_path)[:, :].astype(float)

    with open("letters.txt") as lettersf:
        letters = np.array(list(iter(lettersf.read())))

    def get_letter_at_grid_point(x_index, y_index):
        if (x_index == 0 and y_index == 0) or (x_index == num_of_columns - 1 and y_index == 0) or (x_index == 0 and y_index == num_of_rows - 1) or (
                x_index == num_of_columns - 1 and y_index == num_of_rows - 1):
            return "O"
        if x_index == 0 or y_index == 0 or x_index == num_of_columns - 1 or y_index == num_of_rows - 1:
            return "X"
        if y_index == 1:
            return None
        index = (x_index - 1) + (y_index - 2) * (num_of_columns - 2)
        return letters[index % len(letters)]

    corner_char_image, frame_char_image = get_frame_chars_images()

    # identify corners
    frame_vertices = find_corners_with_char_images(m, corner_char_image, frame_char_image)

    axes_object_to_plot = None

    if b_plot:
        axes_object_to_plot = plt.axes()
        axes_object_to_plot.imshow(m)
        axes_object_to_plot.set_title("examine")

    grid_iterator = grid_iterator_from_corners(
        num_of_rows=num_of_rows, num_of_columns=num_of_columns, quadrilateral_vertices=frame_vertices, partial_cells_overlap=0,
        b_plot=b_plot, axes_object_to_plot=axes_object_to_plot)

    print("collecting character samples:")
    letters_images = {l: [] for l in letters}

    for (x_idx, y_idx), box in grid_iterator:
        cur_letter = get_letter_at_grid_point(x_idx, y_idx)
        if cur_letter is False:
            continue
        x, y = np.meshgrid(np.arange(*box[0]), np.arange(*box[1]), indexing="ij")
        letter_image = np.array(map_coordinates(m, [y, x], order=1, prefilter=False)).T
        letter_image -= np.mean(letter_image)
        letter_image /= np.sum(letter_image ** 2) ** 0.5
        letters_images[cur_letter].append(letter_image)

    # average out all images of each letter perfect letter sample image for each letter
    print("optimizing character samples:")
    letters_images_ideal = {}

    def score_image_sample(arr, c):
        return np.sum([arr * v for v in letters_images[c]]) / len(letters_images[c])

    for c in tqdm.tqdm(letters_images):
        scores = [score_image_sample(arr, c) for arr in letters_images[c]]
        letters_images_ideal[c] = sum([letters_images[c][i] * scores[i] ** 3 for i in range(len(letters_images[c]))])
        letters_images_ideal[c] /= np.sum(letters_images_ideal[c] ** 2) ** 0.5
        if VERBOSITY >= 2:
            print("\'{}\' old best score {:.3g} optimized score {:.3g}".format(c, max(scores), score_image_sample(letters_images_ideal[c], c)))

    with open("Consolas8_600dpi_db.pickle", "wb+") as f:
        pickle.dump(letters_images, f)


def get_frame_chars_images():
    while True:
        try:
            print("please supply a file containing a crop of the corner char of the frame from this scan.")
            corner_char_path = efipy.inquire_input_path()
            corner_char_image = plt.imread(corner_char_path).astype(float)
            break
        except Exception as e:
            print(f"Error: {e}")
    while True:
        try:
            print("please supply a file containing a crop of the frame char of the frame from this scan.")
            frame_char_path = efipy.inquire_input_path()
            frame_char_image = plt.imread(frame_char_path).astype(float)
            break
        except Exception as e:
            print(f"Error: {e}")

    return corner_char_image, frame_char_image


def create_new_calibration_file(output_path,corner_char="O", frame_char="X"):
    font_name = safer_prompt_toolkit.prompt("insert font name", default="ConSolas")
    font_size = safer_prompt_toolkit.prompt("insert font size", default="8")
    page_width = safer_prompt_toolkit.prompt("insert number of characters in row of the page", default="125")
    page_height = safer_prompt_toolkit.prompt("insert number of characters in row of the page", default="77")
    page_width = int(page_width)
    page_height = int(page_height)
    name = "CalibrationFile"
    header = make_file_header(name, font_name, font_size, page_width, page_height)

    with open("letters.txt") as lettersf:
        letters = np.array(list(iter(lettersf.read())))

    page = [""] * page_height
    page.append(header)

    letter_counter = 0
    for row in range(1, page_height):
        for col in range(page_width):
            page[row] += letters[letter_counter]
            letter_counter += 1
        assert len(page[row]) == page_width

    put_page_in_frame(page, page_width, page_height, corner_char, frame_char)

    with open("output_path","w+") as f:
        f.writelines(page)
