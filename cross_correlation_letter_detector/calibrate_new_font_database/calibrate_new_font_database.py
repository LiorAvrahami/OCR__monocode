import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter,rotate,map_coordinates
from scipy.signal import convolve2d #TODO take from improccissing_utilities
import itertools
import pickle
from cross_correlation_letter_detector.grid_iterator import grid_iterator_from_corners

rect_xxyy_of_Ochar = ((181, 217),(170, 119))
num_of_rows = 77 + 2
num_of_columns = 125 + 2
b_plot = True
b_skip = True # TODO remove

m = plt.imread(r"calibration_grayscale_600dpi1.jpg")[:,:].astype(float)

with open("letters.txt") as lettersf:
    letters = np.array(list(iter(lettersf.read())))

def get_letter_at_grid_point(x_index,y_index):
    if (x_index == 0 and y_index == 0) or (x_index == num_of_columns - 1 and y_index == 0) or (x_index == 0 and y_index == num_of_rows - 1) or (x_index == num_of_columns - 1 and y_index == num_of_rows - 1):
        return "O"
    if x_index == 0 or y_index == 0 or x_index == num_of_columns - 1 or y_index == num_of_rows - 1:
        return "X"
    if y_index == 1:
        return False
    index = (x_index - 1) + (y_index - 2)*(num_of_columns-2)
    return letters[index%len(letters)]

if not b_skip:
    ochar = np.copy(m[rect_xxyy_of_Ochar[1][1]:rect_xxyy_of_Ochar[1][0],rect_xxyy_of_Ochar[0][0]:rect_xxyy_of_Ochar[0][1]])
    ochar -= np.mean(ochar)
    ochar /= np.sum(ochar**2)**0.5
    conv = convolve2d(m, ochar, mode="same", boundary="wrap")
    norm = convolve2d(m**2, ochar*0 + 1, mode="same", boundary="wrap")**0.5
    fitv = conv/norm


    def inprod(v1,v2,zoompt2_xy):
        h,w = v1.shape
        rect2 = (int(zoompt2_xy[0] - w/2),int(zoompt2_xy[0] + w/2)),(int(zoompt2_xy[1] - h/2),int(zoompt2_xy[1] + h/2))
        return np.sum(v1*v2[rect2[1][0]:rect2[1][1],rect2[0][0]:rect2[0][1]])

    # identify corners
    thereshold = 0.3
    corners = fitv > thereshold
    wherey,wherex = np.where(corners)
    top_left_index = np.argmin(wherex + wherey)
    top_right_index = np.argmin(-wherex + wherey)
    bottom_left_index = np.argmin(wherex - wherey)
    bottom_right_index = np.argmin(-wherex - wherey)
    top_left_pos = wherex[top_left_index],wherey[top_left_index]
    top_right_pos = wherex[top_right_index],wherey[top_right_index]
    bottom_left_pos = wherex[bottom_left_index],wherey[bottom_left_index]
    bottom_right_pos = wherex[bottom_right_index],wherey[bottom_right_index]
else:
    top_left_pos,top_right_pos,bottom_left_pos,bottom_right_pos = (200-0.5 - 1.6, 144), (4814-2 - 1.6, 147), (213 - 1.6, 6706), (4828-3 - 1.6, 6701)

axes_object_to_plot = None
if b_plot:
    axes_object_to_plot = plt.axes()
    axes_object_to_plot.imshow(m)
    axes_object_to_plot.set_title("examine")

frame_vertices = top_left_pos,top_right_pos,bottom_left_pos,bottom_right_pos

grid_iterator = grid_iterator_from_corners(
    num_of_rows=num_of_rows,num_of_columns=num_of_columns,quadrilateral_vertices=frame_vertices,partial_cells_overlap=0,
    b_plot = b_plot,axes_object_to_plot=axes_object_to_plot)

letters_images = {l:[] for l in letters}

for (x_idx, y_idx),box in grid_iterator:
    cur_letter = get_letter_at_grid_point(x_idx, y_idx)
    if cur_letter is False:
        continue
    x, y = np.meshgrid(np.arange(*box[0]), np.arange(*box[1]), indexing="ij")
    letter_image = map_coordinates(m, [y, x], order=1, prefilter=False).T
    letter_image -= np.mean(letter_image)
    letter_image /= np.sum(letter_image ** 2) ** 0.5
    letters_images[cur_letter].append(letter_image)

# with open("Consolas8_600dpi_db.pickle","wb+") as f:
#     pickle.dump(letters_images,f)
