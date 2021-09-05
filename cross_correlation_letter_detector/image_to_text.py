import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter,rotate,map_coordinates
from .improccissing_utilities import correlate, normalise_kernel_in_place, calculate_normalised_crosscorrelation
import itertools
import pickle
from .grid_iterator import grid_iterator_from_corners
import tqdm

b_rotate = False
partial_cells_overlap = -0.5 # should be between 0 and 3, make it big if text is not very grid-like,
b_plot_grid = False

with open("Consolas8_600dpi_db.pickle","rb") as f:
    letters_images = pickle.load(f)

document_image = plt.imread(r"calibration_grayscale_600dpi1.jpg")[:, :].astype(float)

Xidle = np.mean(letters_images['X'][10:20],axis=0)
fitX = calculate_normalised_crosscorrelation(document_image,Xidle)
fitX_g = gaussian_filter(fitX**4,5)

def get_two_largest_local_maxima(profile):
    temp = abs(profile[1:-1] * np.diff(np.sign(np.diff(profile))))
    idx1 = np.argmax(temp)
    temp[idx1] = 0
    idx2 = np.argmax(temp)
    return min(idx1,idx2),max(idx1,idx2)

# rotate TODO

# find frame
x_profile = np.mean(fitX_g,axis=1)
framexL,framexR = get_two_largest_local_maxima(x_profile)

y_profile = np.mean(fitX_g,axis=0)
frameyU,frameyD = get_two_largest_local_maxima(y_profile)

frame_vertices = (frameyU,framexL),(frameyD,framexL),(frameyU,framexR),(frameyD,framexR)

axes_to_plot = None
if b_plot_grid:
    axes_to_plot = plt.axes()
    axes_to_plot.imshow(document_image)
    axes_to_plot.set_title("examine")

# find num of rows, columnes TODO
num_of_rows = 77
num_of_columns = 125

letters = np.array(list(letters_images.keys()))
grid_letter_match_results = np.full((num_of_columns,num_of_rows,len(letters),),np.nan,np.float)
for letter_index, letter_char in enumerate(tqdm.tqdm(letters,position=0)):
    letter_image = letters_images[letter_char][0]
    char_fit_arr = calculate_normalised_crosscorrelation(document_image,letter_image)

    grid_iterator = grid_iterator_from_corners(
        num_of_rows=num_of_rows + 2, num_of_columns=num_of_columns + 2, quadrilateral_vertices=frame_vertices,
        partial_cells_overlap=partial_cells_overlap,
        b_plot=b_plot_grid, axes_object_to_plot=axes_to_plot)

    for (x_idx, y_idx),box in grid_iterator:
        if x_idx == 0 or y_idx == 0 or x_idx == num_of_columns + 1 or y_idx == num_of_rows + 1:
            continue
        # if y_idx == 1 or y_idx > 2 or x_idx > 20:#todo remove
        #     continue

        grid_cell_char_fit_arr = char_fit_arr[int(box[1][0]):int(np.ceil(box[1][1])),int(box[0][0]):int(np.ceil(box[0][1]))]
        grid_letter_match_results[x_idx - 1, y_idx - 1, letter_index] = np.max(grid_cell_char_fit_arr)

result_text = letters[np.argmax(grid_letter_match_results,axis=-1)].T
result_text = "\n".join(["".join(ln) for ln in result_text])

np.savez("ocr-results",result_text,grid_letter_match_results,letters)

with open("text_temp.txt", "w+") as f:
    f.write(result_text)



