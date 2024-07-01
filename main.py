# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# The main code script for this project.
# Contains a number of functions essential for the functionality
# exposed in run.py, and utilizes most of the tools in this
# project to achieve that functionality.

import GAT.anacombetransform
from Adaptive_PCA import vectorize, devectorize, clustering, adaptive_clustering
from Wiener_filter import lpa_ici, wiener_filter
from Zero_mean.zero_mean import zero_mean
from Correlation.correlationenergy import peak_correlation_energy, signed_peak_correlation_energy
from Remove_diagonal.wavelet import remove_diagonal

from globals import large_window_size, mu, max_thread_count, coefficients_sigma, peak_size

from multiprocessing import Pool
import numpy as np

def denoise(image):
    """Performs the main denoising step, including the GAT, clustering steps,
    PCA filtering and suboptimal Wiener filter.
    This operation works on a 2d array, so one color channel at the time."""
    Image_v = GAT.anacombetransform.anacombe(image)

    v_windows, h_windows = Image_v.shape
    v_windows = (v_windows + large_window_size - 1) // large_window_size
    h_windows = (h_windows + large_window_size - 1) // large_window_size

    reconstructed_image = np.zeros_like(Image_v)
    image_counts = np.zeros_like(Image_v)

    # Make 128x128 non-overlapping windows
    for base_x in range(h_windows):
        for base_y in range(v_windows):
            print(f"Processing large window x: {base_x+1}/{h_windows}, y: {base_y+1}/{v_windows}")

            blocks = vectorize.vectorize(Image_v[base_y * large_window_size:base_y * large_window_size + large_window_size,base_x * large_window_size:base_x * large_window_size + large_window_size], large_window_size)

            clusters = clustering.k_means(blocks, 3, 4, True)
            reclustered = []
            for cluster in clusters:
                mergeable_clusters = [clustering.Cluster(blocks[i], [i]) for i in cluster.indices]
                reclustered.extend(adaptive_clustering.clustering(mergeable_clusters))

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

                extracted_coefficients = xi[:dominant_dimentions]

                filtered_coefficients = wiener_filter.wiener_filter(extracted_coefficients)

                Ns = np.sqrt(Na * filtered_coefficients)
                # Recompose matrix.
                Ar = np.dot(U[:,:dominant_dimentions] * Ns, Vh[:dominant_dimentions,:])

                # reconstruct the small window, add it to the reconstructed image.
                large_window_recon, large_window_samples = devectorize(Na, cluster, Ar)

                global_x_start, global_y_start = base_x * large_window_size, base_y * large_window_size
                global_x_end, global_y_end = global_x_start + large_window_size, global_y_start + large_window_size

                reconstructed_image[global_y_start : global_y_end, global_x_start : global_x_end] += large_window_recon
                image_counts[global_y_start : global_y_end, global_x_start : global_x_end] += large_window_samples



    reconstructed_image = reconstructed_image / image_counts

    W = GAT.anacombetransform.inv_anacombe(reconstructed_image)
    return W

def denoise_full(image_tuple):
    """Calculate PRNU noise residue.
    Takes in a tuple (i, image) where i is an int, image is an image saved as a [y,x,3]-shaped np array.
    Then returns a tuple (i, image, residue), where residue is the extracted noise.
    This operation applies both the primary denoising algorithm, as well as two improvement steps,
    to all 3 channels of an image.
    As a result, this function can take a couple hours for larger images."""

    i, image = image_tuple
    residue = np.empty_like(image)

    for channel in range(3):
        print(f"Processing channel {channel} of image {i}.")
        residue[:,:,channel] = image[:,:,channel] - denoise(image[:,:,channel])

    residue_ZM = zero_mean(residue)
    residue_ZM_RD = remove_diagonal(residue_ZM)
    return (i, image, residue_ZM_RD)


def extract_residues(images, outpath, multithreaded = True):
    """Extracts the PRNU residue of a large number of images
    in separate processes."""

    if multithreaded:
        with Pool(min(max_thread_count, len(images))) as p:
            for (i, _, residue) in p.imap_unordered(denoise_full, enumerate(images)):
                if isinstance(outpath, str):
                    usepath = outpath + f'{i}.npy'
                else:
                    usepath = outpath[i]
                with open( usepath, 'wb') as f:
                    np.save(f, residue)
    else:
        for (i, _, residue) in map(denoise_full, enumerate(images)):
            if isinstance(outpath, str):
                usepath = outpath + f'{i}.npy'
            else:
                usepath = outpath[i]
            with open( usepath, 'wb') as f:
                np.save(f, residue)
    

def find_fingerprint(images, residues):
    """Calculates the image fingerprint when given a large number
    of images and their corresponding PRNU noise residues."""
    top = np.zeros_like(images[0], dtype='float64')
    bottom = np.zeros_like(images[0], dtype='float64')
    assert len(images) == len(residues)
    for i in range(len(images)):
            top += np.multiply(images[i], residues[i])
            bottom += np.multiply(images[i], images[i])

    K = np.divide(top, bottom, out=np.zeros_like(top), where=(bottom!=0))
    fingerprint = np.sum(np.multiply(K, image) for image in images) / len(images)
    return fingerprint

def test_fingerprint_PCE_multiple(fingerprint, residues, names):
    """Compare a series of residues against the fingerprint."""
    assert len(residues) == len(names)
    for (residue, name) in zip(residues, names):
        pce_red = peak_correlation_energy(residue, fingerprint, peak_size, 0)
        pce_green = peak_correlation_energy(residue, fingerprint, peak_size, 1)
        pce_blue = peak_correlation_energy(residue, fingerprint, peak_size, 2)
        print(f"Image {name} has a PCE of {pce_red}, {pce_green}, {pce_blue}")

def test_fingerprint_SPCE_multiple(fingerprint, residues, names):
    """Compare a series of residues against the fingerprint."""
    assert len(residues) == len(names)
    for (residue, name) in zip(residues, names):
        spce_red = signed_peak_correlation_energy(residue, fingerprint, peak_size, 0)
        spce_green = signed_peak_correlation_energy(residue, fingerprint, peak_size, 1)
        spce_blue = signed_peak_correlation_energy(residue, fingerprint, peak_size, 2)
        print(f"Image {name} has a SPCE of {spce_red}, {spce_green}, {spce_blue}")
