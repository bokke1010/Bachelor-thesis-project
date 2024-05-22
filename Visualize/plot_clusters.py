
from matplotlib import pyplot as plt

# Take clusters to be [clustering.Cluster]

def pitc(imsize):
    def indextocoord(i):
        slen = imsize // 2 - 3
        return (2 * (i % slen)+3, 2 * (i // slen)+3)
    return indextocoord

# clustering image
def plot_clusters(base_image_channel, imsize, clusters):
    plt.imshow(base_image_channel[0:imsize,0:imsize], cmap="gray")

    voxels = list(map(pitc(imsize), clusters[0].indices))
    plt.scatter(*zip(*voxels), c="b", s=0.9, marker="s")
    voxels = list(map(pitc(imsize), clusters[1].indices))
    plt.scatter(*zip(*voxels), c="r", s=0.9, marker="s")
    voxels = list(map(pitc(imsize), clusters[2].indices))
    plt.scatter(*zip(*voxels), c="g", s=0.9, marker="s")
    plt.show()

# clustering hist
def plot_histogram(clusters):
    plt.hist([len(x.indices) for x in clusters], bins=list(2**x - 0.5 for x in range(11)))
    plt.xscale('log')
    plt.show()

# clustering image
def plot_nth_biggest_cluster(base_image_channel, imsize, clusters, cluster_index):
    plt.imshow(base_image_channel[0:imsize,0:imsize], cmap="gray")
    
    biggestcluster = sorted(clusters, key=lambda c : len(c.indices), reverse=True)[cluster_index]
    
    voxels = list(map(pitc(imsize), biggestcluster.indices))
    plt.scatter(*zip(*voxels), s=0.9, marker="s")
    plt.show()