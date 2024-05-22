# import sklearn
import numpy as np

window_size = 8

# M = sum sigma_i u_i v_i*
# v* is the conjugate transpose
def vectorize(large_window: np.array, large_window_size):
    # A_M is a C times N_A sized matrix, where C is window_size (or squared?)
    # and N_A is the number of individuals (small windows?) in cluster A
    blockcount = (large_window_size - window_size + 2) // 2
    blocks = np.empty((blockcount**2, window_size**2))
    for y in range(0, large_window_size - window_size + 1, 2):
        for x in range(0, large_window_size - window_size + 1, 2):
            window = large_window[y:y+window_size,x:x+window_size]
            i = (y * blockcount + x) // 2
            blocks[i] = window.flatten()
    return blocks


