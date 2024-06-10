from random import sample
import numpy as np

class Cluster:
    center = None
    def __init__(self, center, indices = []):
        self.center = center
        self.indices = indices
    def __repr__(self):
        return f"Cluster at {repr(self.center)} containing indices {repr(self.indices)}."
    def merge(self, other):
        selflen, otherlen = len(self.indices), len(other.indices)
        self.center = (self.center * selflen + other.center * otherlen) / (selflen + otherlen)
        self.indices = self.indices + other.indices
        # self.indices.extend(other.indices)

def k_means(vectors, k, n, ret_clusters=False):
    # new sum, count, old avg
    vlen = vectors.shape[1]
    means = vectors[np.random.choice(vectors.shape[0], k, replace=False), :]
    data = np.zeros(shape=(k, vlen))
    count = np.zeros(k)
    for _ in range(n):
        # Find nearest cluster
        for vector in vectors:
            i = np.argmin(np.sum(np.square(means - vector), axis=1))
            data[i] += vector
            count[i] += 1

        means = data / count[:,None]
        data.fill(0)
        count.fill(0)
    if not ret_clusters:
        return means
    
    clusters = [Cluster(means[i], []) for i in range(k)]
    for (i, vector) in enumerate(vectors):
        m = np.argmin(np.sum(np.square(means - vector), axis=1))
        clusters[m].indices.append(i)
    return clusters
    
    