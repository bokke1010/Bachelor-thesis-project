# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# A simple script takes a 2d array of size [large_window_size]
# and returns a big array of overlapping 8x8 small windows with
# stride 2

import numpy as np
from globals import window_size, window_stride, large_window_size


# M = sum sigma_i u_i v_i*
# v* is the conjugate transpose
def vectorize(large_window: np.array, large_window_size):
    # A_M is a C times N_A sized matrix, where C is window_size squared
    # and N_A is the number of individuals small windows in cluster A
    blockcount = (large_window_size - window_size + 2) // 2
    blocks = np.empty((blockcount**2, window_size**2))
    for y in range(0, large_window_size - window_size + 1, 2):
        for x in range(0, large_window_size - window_size + 1, 2):
            window = large_window[y:y+window_size,x:x+window_size]
            i = (y * blockcount + x) // 2
            blocks[i] = window.flatten()
    return blocks


def devectorize(size, position_reference, matrix):
    reconstructed_image = np.zeros((large_window_size, large_window_size), dtype='float64')
    image_counts = np.zeros((large_window_size, large_window_size), dtype='float64')
    for column in range(size):
        block_index = position_reference.indices[column]
        block = matrix[column].reshape((window_size,window_size))
        block_count = (large_window_size - window_size + window_stride) // window_stride

        block_x_index, block_y_index = block_index % block_count, block_index // block_count
        block_x_offset, block_y_offset = block_x_index * window_stride, block_y_index * window_stride

        reconstructed_image[block_y_offset : block_y_offset + window_size, block_x_offset : block_x_offset + window_size] += block
        image_counts[block_y_offset : block_y_offset + window_size, block_x_offset : block_x_offset + window_size] += 1
    return (reconstructed_image, image_counts)
