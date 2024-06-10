import GAT.anacombetransform
from Adaptive_PCA import vectorize, clustering, adaptive_clustering
from Wiener_filter import lpaici, wiener_filter
from Zero_mean.zero_mean import zero_mean
from tools.np_imageload import load_image, save_image, save_image_grayscale
from Remove_diagonal.wavelet import remove_diagonal

import numpy as np

use_multiprocessing = True
if use_multiprocessing:
    from multiprocessing import Pool

large_window_size = 128
window_size = 8
window_stride = 2

# I = Poisson(real_I) + N(0, sigma)
# Estimated image normal noise deviation
sigma = 2
GAT.anacombetransform.sigma = sigma

# The largest allowed distance between clusters for them to be merged.
clustering_threshold = 3000

# Noise dimention threshold multiplier used in determining
# dominant dimensions for PCA filtering.
mu = 1.1

# Adaptive clustering coefficients to more
# easily cluster large clusters together.
adaptive_clustering.kappa = 0.7

# Confidence interval theshold parameter
tau = 0.6
lpaici.tau = tau

# These are the same sigma, and separate from the image model sigma
coefficients_sigma = 0.8
lpaici.sigma = coefficients_sigma
wiener_filter.sigma = coefficients_sigma

def denoise(image):
    # Expand image 8 pixels somehow?

    Image_v = GAT.anacombetransform.anacombe(image)

    v_windows, h_windows = Image_v.shape
    v_windows = (v_windows + large_window_size - 1) // large_window_size
    h_windows = (h_windows + large_window_size - 1) // large_window_size
    # Make 128x128 non-overlapping windows
    # make 8x8 stride 2 windows
    # Vectorize overlapping blocks

    reconstructed_image = np.zeros_like(Image_v)
    image_counts = np.zeros_like(Image_v)

    for base_x in range(h_windows):
        for base_y in range(v_windows):
            print(f"Processing large window {base_x+1}/{h_windows}, {base_y+1}/{v_windows}.")

            blocks = vectorize.vectorize(Image_v[base_y * large_window_size:base_y * large_window_size + large_window_size,base_x * large_window_size:base_x * large_window_size + large_window_size], large_window_size)

            clusters = clustering.k_means(blocks, 3, 4, True)

            reclustered = []
            for cluster in clusters:
                mergeable_clusters = [clustering.Cluster(blocks[i], [i]) for i in cluster.indices]
                reclustered.extend(adaptive_clustering.clustering(mergeable_clusters, clustering_threshold))

            for cluster in reclustered:
                Am = np.stack([blocks[i] for i in cluster.indices])
                (Na, C) = Am.shape
                U, S, Vh = np.linalg.svd(Am)

                # s = sqrt(Na xi_k) => xi_k = s^2 / Na
                xi = S**2 / Na
                gamma = C / Na
                xi_r = coefficients_sigma**2 * (1 + np.sqrt(gamma))**2

                dominant_dimentions = 0
                for i in range(len(xi)):
                    if xi[i] < mu * xi_r:
                        break
                    dominant_dimentions += 1
                # dominant_dimentions = sum(xi[k] > mu * xi_r for k in range(min(C,Na)))

                extracted_coefficients = xi[:dominant_dimentions]

                filtered_coefficients = wiener_filter.wiener_filter(extracted_coefficients)

                Ns = np.sqrt(Na * filtered_coefficients)
                # Recompose matrix.
                Ar = np.dot(U[:,:dominant_dimentions] * Ns, Vh[:dominant_dimentions,:])

                for column in range(Na):
                    block_index = cluster.indices[column]
                    block = Ar[column].reshape((window_size,window_size))
                    blockcount = (large_window_size - window_size + window_stride) // window_stride
                    # i = (y * blockcount + x) // window_stride

                    block_y_start = block_index // blockcount
                    block_x_start = block_index % blockcount
                    global_x_start = base_x * large_window_size + block_x_start * window_stride
                    global_y_start = base_y * large_window_size + block_y_start * window_stride
                    reconstructed_image[global_y_start : global_y_start + window_size, global_x_start : global_x_start + window_size] += block
                    image_counts[global_y_start : global_y_start + window_size, global_x_start : global_x_start + window_size] += 1


    reconstructed_image = reconstructed_image / image_counts

    W = GAT.anacombetransform.inv_anacombe(reconstructed_image)
    return W

def denoise_full(image):
    residue = np.empty_like(image)

    for channel in range(3):
        residue[:,:,channel] = image[:,:,channel] - denoise(image[:,:,channel])

    residue_ZM = zero_mean(residue)
    residue_ZM_RD = remove_diagonal(residue_ZM)
    return (image, residue_ZM_RD)


def find_fingerprint(images):
    top = np.zeros_like(images[0])
    bottom = np.zeros_like(images[0])

    with Pool(14) as p:
        for (image, residue) in p.imap_unordered(denoise_full, images):
            top += np.multiply(image, residue)
            bottom += np.multiply(image, image)
    K = top / bottom
    fingerprint = np.sum(np.multiply(K, image) for image in images) / len(images)
    save_image(fingerprint + 128)
    # for i, image in enumerate(images):


if __name__ == '__main__':
    image = load_image("muis_small.png")
    
    find_fingerprint([image])
