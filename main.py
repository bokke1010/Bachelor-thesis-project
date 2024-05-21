import GAT.anacombetransform
from Adaptive_PCA import vectorize, clustering, adaptive_clustering

from matplotlib import pyplot as plt

from PIL import Image
import numpy as np
from scipy import ndimage

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )

sigma = 2

image = load_image("muis_small.png")
red_channel = image[:,:,0]

GAT.anacombetransform.sigma = sigma
Image_v = GAT.anacombetransform.anacombe(red_channel)

# Make 128x128 non-overlapping windows
# make 8x8 stride 2 windows
# Vectorize overlapping blocks?
# 

blocks = vectorize.vectorize(Image_v[0:128,0:128])

clusters = clustering.k_means(blocks, 3, 4, True)
# print(sum(len(cluster.indices) for cluster in clusters), len(blocks))

# clustering image
# plt.imshow(Image_v[0:128,0:128], cmap="gray")
# def indextocoord(i):
#     slen = len(blocks) ** 0.5
#     assert slen == int(slen)
#     return (2 * (i % slen)+3, 2 * (i // slen)+3)
# voxels = list(map(indextocoord, clusters[0].indices))
# plt.scatter(*zip(*voxels), c="b", s=0.9, marker="s")
# voxels = list(map(indextocoord, clusters[1].indices))
# plt.scatter(*zip(*voxels), c="r", s=0.9, marker="s")
# voxels = list(map(indextocoord, clusters[2].indices))
# plt.scatter(*zip(*voxels), c="g", s=0.9, marker="s")
# plt.show()

reclustered = []
for cluster in clusters:
    mergeable_clusters = [clustering.Cluster(blocks[i], [i]) for i in cluster.indices]

    reclustered.extend(adaptive_clustering.clustering(mergeable_clusters, 2000))
    #TODO: visualize this step

# clustering hist
# plt.hist([len(x.indices) for x in reclustered], bins=list(2**x - 0.5 for x in range(11)))
# plt.xscale('log')
# plt.show()

# clustering image
# plt.imshow(Image_v[0:128,0:128], cmap="gray")
# def indextocoord(i):
#     slen = len(blocks) ** 0.5
#     assert slen == int(slen)
#     return (2 * (i % slen)+3, 2 * (i // slen)+3)
# # biggestcluster = max(reclustered, key=lambda c : len(c.indices))
# biggestcluster = sorted(reclustered, key=lambda c : len(c.indices), reverse=True)[1]
# print(len(biggestcluster.indices))
# voxels = list(map(indextocoord, biggestcluster.indices))
# plt.scatter(*zip(*voxels), s=0.9, marker="s")
# plt.show()

# Noise dimention threshold multiplier
mu = 1.1

for cluster in reclustered:
    Am = np.stack([blocks[i] for i in cluster.indices])
    (Na, C) = Am.shape
    U, S, Vh = np.linalg.svd(Am)

    # s = sqrt(Na xi_k) => xi_k = s^2 / Na
    xi = S**2 / Na
    gamma = C / Na
    xi_r = sigma**2 * (1 + np.sqrt(gamma))**2

    R = sum(xi[k] > mu * xi_r for k in range(min(C,Na)))

    coeffs = S[:R]
    tau = 0.6
    def CIe(x, h):
        window = list(range(x-h, x+h+1))
        Nh = len(window)
        Cxh = sum(coeffs[s] for s in window)/Nh
        stdxh = sigma / np.sqrt(Nh)
        return (Cxh - stdxh, Cxh + stdxh)

    def ICI(x, hvals):
        hmax = hvals[0]
        (L, U) = CIe(x,hmax)
        for h in hvals[1:]:
            (Ln, Un) = CIe(x,h)
            if max(Ln, L) < min(Un, U):
                break
            L, U = max(Ln, L), min(Un, U)
            hmax = h
        return hmax
    
    def autocovar(x, h):
        window = list(range(x-h, x+h+1))
        return sum(coeffs[s]**2 for s in window) / len(window)
    
    def maxAttenuation(sigma, Rx, beta = 0.7):
        # Find attenuation in [0, 1] so that
        def costfunction(atten):
            sq = sigma**2
            hs = 1 - atten
            g0 = 1 - atten * sq / Rx
            gRg = g0**2 * Rx
            return (sq + gRg - 2 * sq * g0) / (sq + atten**2 * gRg - 2 * atten * sq * g0) - beta * atten**2
        best_atten = 0
        best_result = -10e6
        # Todo: sensible, accurate approximation
        # Current technique is slow and up to 0.005 away from the optimal value
        for atten in np.linspace(0,1,100):
            cost = costfunction(atten)
            if cost > best_result:
                best_result = cost
                best_atten = atten
        return best_atten


    newcoeffs = np.empty_like(coeffs)
    for x in range(R):
        bandwidth = ICI(x, range(10))
        Rx = autocovar(x, bandwidth)

        attenuation = maxAttenuation(sigma, Rx)
        hs = 1 - attenuation
        newcoeffs[x] = hs * coeffs[x]
    Ns = np.sqrt(Na * newcoeffs)
    # Recompose matrix.
    # Ar = np.dot(U[:, :R] * S[:R,:R], Vh[:R,:])






# for each clusters do
#     10: Dimension Selection Based on PCA Eigenvalue Hard Threshold
#     11: for coefficient in each selected dimension do
#         12: Estimate the parameter of filter with LPA-ICI
#         13: Suboptimal Wiener filter denoising
#     14: end for
# 15: end for


# Image_vd = ndimage.gaussian_filter(Image_v, sigma)

# W = GAT.anacombetransform.inv_anacombe(Image_vd)

# image[:,:,0] = completed
# save_image(10 * np.abs(W - red_channel), "difference.jpg")
# save_image(W, "modified.jpg")