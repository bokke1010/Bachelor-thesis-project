import Adaptive_PCA.clustering as cl
from random import random
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np

vectors = np.random.rand(64, 2)
means = cl.k_means(vectors, 3, 10)

from matplotlib import pyplot as plt

vor = Voronoi(means)
fig = voronoi_plot_2d(vor)
fig.gca().scatter(vectors[:,0], vectors[:,1], c="green")
fig.gca().scatter(means[:,0], means[:,1], c="red")
fig.gca().set_xlim(xmin=0,xmax=1)
fig.gca().set_ylim(ymin=0,ymax=1)

plt.show()