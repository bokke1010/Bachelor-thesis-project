
# Large window size must be even, and neatly divide both the
# horizontal and vertical resolution of the images.
large_window_size = 252
window_size = 8
window_stride = 2
max_thread_count = 12

# I = Poisson(real_I) + N(0, sigma)
# Estimated image normal noise deviation
sigma = 2

# The largest allowed distance between clusters for them to be merged.
clustering_threshold = 320

# Noise dimention threshold multiplier used in determining
# dominant dimensions for PCA filtering.
mu = 1.1

# Adaptive clustering coefficients to more
# easily cluster large clusters together.
kappa = 0.7

# Confidence interval theshold parameter
tau = 0.6

# Singular value based coefficient noise deviation
coefficients_sigma = 0.8


# Size of the peak for calculating PCE and SPCE.
# Uses euclidean distance
peak_size = 3
