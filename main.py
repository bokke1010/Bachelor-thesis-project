import GAT.anacombetransform
from Adaptive_PCA import vectorize, clustering, adaptive_clustering
from Wiener_filter import lpaici, wiener_filter
from Zero_mean.zero_mean import zero_mean
from tools.stat import crosscorr
from Correlation.correlationenergy import peak_correlation_energy, signed_peak_correlation_energy
from Remove_diagonal.wavelet import remove_diagonal
from multiprocessing import Pool

import numpy as np

large_window_size = 252
window_size = 8
window_stride = 2
max_thread_count = 13

# I = Poisson(real_I) + N(0, sigma)
# Estimated image normal noise deviation
sigma = 2
GAT.anacombetransform.sigma = sigma

# The largest allowed distance between clusters for them to be merged.
clustering_threshold = 320

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
            print(f"Processing large window x: {base_x+1}/{h_windows}, y: {base_y+1}/{v_windows}")

            blocks = vectorize.vectorize(Image_v[base_y * large_window_size:base_y * large_window_size + large_window_size,base_x * large_window_size:base_x * large_window_size + large_window_size], large_window_size)

            clusters = clustering.k_means(blocks, 3, 4, True)
            reclustered = []
            for cluster in clusters:
                mergeable_clusters = [clustering.Cluster(blocks[i], [i]) for i in cluster.indices]
                reclustered.extend(adaptive_clustering.clustering(mergeable_clusters, clustering_threshold))

            for cluster in reclustered:
                Am = np.stack(blocks[cluster.indices])
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

def denoise_full(image_tuple):
    """Calculate PRNU noise residue.
    Takes in a tuple (i, image) where i is an int, image is an image saved as a [y,x,3]-shaped np array.
    Then returns a tuple (i, image, residue), where residue is the extracted noise."""

    i, image = image_tuple
    residue = np.empty_like(image)

    for channel in range(3):
        print(f"Processing channel {channel} of image {i}.")
        residue[:,:,channel] = image[:,:,channel] - denoise(image[:,:,channel])

    residue_ZM = zero_mean(residue)
    residue_ZM_RD = remove_diagonal(residue_ZM)
    return (i, image, residue_ZM_RD)


def extract_residues(images, outpath, multithreaded = True):
    top = np.zeros_like(images[0], dtype='float64')
    bottom = np.zeros_like(images[0], dtype='float64')

    if multithreaded:
        with Pool(min(max_thread_count, len(images))) as p:
            for (i, image, residue) in p.imap_unordered(denoise_full, enumerate(images)):
                if isinstance(outpath, str):
                    usepath = outpath + f'{i}.npy'
                else:
                    usepath = outpath[i]
                with open( usepath, 'wb') as f:
                    np.save(f, residue)
    else:
        for (i, image, residue) in map(denoise_full, enumerate(images)):
            if isinstance(outpath, str):
                usepath = outpath + f'{i}.npy'
            else:
                usepath = outpath[i]
            with open( usepath, 'wb') as f:
                np.save(f, residue)
    

def find_fingerprint(images, residues):
    top = np.zeros_like(images[0], dtype='float64')
    bottom = np.zeros_like(images[0], dtype='float64')
    assert len(images) == len(residues)
    for i in range(len(images)):
            top += np.multiply(images[i], residues[i])
            bottom += np.multiply(images[i], images[i])

    K = np.divide(top, bottom, out=np.zeros_like(top), where=(bottom!=0))
    fingerprint = np.sum(np.multiply(K, image) for image in images) / len(images)
    return fingerprint

def normalizing_factor(fingerprint, residue):
    #TODO: figure this out
    pass

def test_fingerprint_SPE(fingerprint, residue):
    # How to determine omega
    # Just assume a 7x7 region around centerpoint
    Omega = [(a, b) for a in range(-3, 4) for b in range(-3, 4)]
    return peak_correlation_energy(fingerprint, residue, Omega)

def test_fingerprint_SPCE(fingerprint, residue):
    return signed_peak_correlation_energy(fingerprint, residue, [(0,0)])
