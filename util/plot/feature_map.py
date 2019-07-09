import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .utils import *


def plot_conv_output(conv_img, plot_dir, name, filters_all=True, filters=[0]):
    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters
    if filters_all:
        num_filters = conv_img.shape[3]
        filters = range(conv_img.shape[3])
    else:
        num_filters = len(filters)

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate filters
    if num_filters == 1:
        img = conv_img[0, :, :, filters[0]]
        axes.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap=cm.hot)
        # remove any labels from the axes
        axes.set_xticks([])
        axes.set_yticks([])
    else:
        for l, ax in enumerate(axes.flat):
            # get a single image
            img = conv_img[0, :, :, filters[l]]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap=cm.hot)
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
    # save figure
    print(os.path.join(plot_dir, '{}.png'.format(name)))
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')


