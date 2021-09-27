import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter,rotate,map_coordinates
from scipy.signal import convolve2d
import itertools
import pickle

def grid_iterator_from_corners(num_of_rows,num_of_columns,quadrilateral_vertices,partial_cells_overlap = 0,b_plot=False,axes_object_to_plot=None):
    """
    :param num_of_rows,num_of_columns: number of rows and number of columns in grid
    :param quadrilateral_vertices: the vertices of the quadrilateral that will be the boundary of the grid.
    :param partial_cells_overlap: used to iterate over overlapping grid cells, the grid cells that are returned
    are expand so that their volume increases by partial_cells_overlap * origenal volume, which is to say that
    their volume increases by a factor of (1 + partial_cells_overlap). defaults to 0
    :param b_plot: weather to plot the grid points before iterating on them ( for looking at grid before iterating over it )
    :param axes_object_to_plot: where to plot the grid (only relevant if b_plot is true), if None than a new axis is crated.
    :return:
    """
    top_left_pos, top_right_pos, bottom_left_pos, bottom_right_pos = quadrilateral_vertices
    # define grid
    part_x = lambda x: x / (num_of_columns - 1)
    part_y = lambda y: y / (num_of_rows - 1)
    x, y = np.meshgrid(range(num_of_columns), range(num_of_rows), indexing="ij")
    gridx = top_left_pos[0] * (1 - part_x(x)) * (1 - part_y(y)) + \
            top_right_pos[0] * (part_x(x)) * (1 - part_y(y)) + \
            bottom_left_pos[0] * (1 - part_x(x)) * (part_y(y)) + \
            bottom_right_pos[0] * (part_x(x)) * (part_y(y))
    gridy = top_left_pos[1] * (1 - part_x(x)) * (1 - part_y(y)) + \
            top_right_pos[1] * (part_x(x)) * (1 - part_y(y)) + \
            bottom_left_pos[1] * (1 - part_x(x)) * (part_y(y)) + \
            bottom_right_pos[1] * (part_x(x)) * (part_y(y))

    dx = np.gradient(gridx, axis=0)  # I do it this way because it seems more readable & less prone to bugs
    dy = np.gradient(gridy, axis=1)  # I do it this way because it seems more readable & less prone to bugs

    if b_plot:
        if axes_object_to_plot is None:
            axes_object_to_plot = plt.axis()
        axes_object_to_plot.plot([top_left_pos[0], top_right_pos[0], bottom_left_pos[0], bottom_right_pos[0]],
                 [top_left_pos[1], top_right_pos[1], bottom_left_pos[1], bottom_right_pos[1]], "or", alpha=0.5)
        axes_object_to_plot.plot(gridx.flatten(), gridy.flatten(), "om", alpha=0.3)
        for (_,_), box in grid_iterator_from_corners(num_of_rows,num_of_columns,quadrilateral_vertices,partial_cells_overlap = partial_cells_overlap,b_plot=False,axes_object_to_plot=axes_object_to_plot):
            xb, yb = box
            plt.plot([xb[0],xb[1],xb[1],xb[0],xb[0]],[yb[0],yb[0],yb[1],yb[1],yb[0]],"b")
        plt.show()

    for y_idx,x_idx in itertools.product(range(num_of_rows),range(num_of_columns)):
        box = (gridx[x_idx, y_idx] - dx[x_idx,y_idx]*(1 + partial_cells_overlap)**0.5 / 2, gridx[x_idx, y_idx] + dx[x_idx,y_idx]*(1 + partial_cells_overlap)**0.5 / 2),\
              (gridy[x_idx, y_idx] - dy[x_idx,y_idx]*(1 + partial_cells_overlap)**0.5 / 2, gridy[x_idx, y_idx] + dy[x_idx,y_idx]*(1 + partial_cells_overlap)**0.5 / 2)
        yield (x_idx, y_idx),box