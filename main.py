import GAT.anacombetransform
from Adaptive_PCA import vectorize, clustering, adaptive_clustering
from Wiener_filter import lpaici, wiener_filter
from zero_mean import zero_mean

from PIL import Image
import numpy as np
from multiprocessing import Pool

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "RGB" )
    img.save( outfilename )

def save_image_grayscale( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )

large_window_size = 128
window_size = 8
window_stride = 2

# I = Poisson(real_I) + N(0, sigma)
# Estimated image normal noise deviation
sigma = 2

# The largest allowed distance between clusters for them to be merged.
clustering_threshold = 3000

# Noise dimention threshold multiplier used in determining
# dominant dimensions for PCA filtering.
mu = 1.1

# Confidence interval theshold parameter
tau = 0.6

# Adaptive clustering coefficients to more
# easily cluster large clusters together.
adaptive_clustering.kappa = 0.7

GAT.anacombetransform.sigma = sigma
lpaici.sigma = sigma
lpaici.tau = tau
wiener_filter.sigma = sigma

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
            print(len(reclustered))

            for cluster in reclustered:
                Am = np.stack([blocks[i] for i in cluster.indices])
                (Na, C) = Am.shape
                U, S, Vh = np.linalg.svd(Am)

                # s = sqrt(Na xi_k) => xi_k = s^2 / Na
                xi = S**2 / Na
                gamma = C / Na
                xi_r = sigma**2 * (1 + np.sqrt(gamma))**2

                R = sum(xi[k] > mu * xi_r for k in range(min(C,Na)))

                extracted_coefficients = xi[:R]

                filtered_coefficients = wiener_filter.wiener_filter(extracted_coefficients)


                Ns = np.sqrt(Na * filtered_coefficients)
                # Recompose matrix.
                Ar = np.dot(U[:,:R] * Ns, Vh[:R,:])

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


if __name__ == '__main__':
    # image = load_image("Sprite-0001.png")
    image = load_image("muis_small.png")
    clean_image = np.empty_like(image)
    residue = np.empty_like(image)
    denoised = None
    with Pool(3) as p:
        denoised = p.map(denoise, [image[:,:,channel] for channel in range(3)])

    for channel in range(3):
        current_channel = image[:,:,channel]
        clean_image[:,:,channel] = denoised[channel]
        residue[:,:,channel] = current_channel - denoised[channel]

    save_image(clean_image, "denoised.png")
    # save_image(image, "denoised_full.png")
    # residue_ZM = zero_mean(residue)
    save_image(128 + residue, "noise.png")
