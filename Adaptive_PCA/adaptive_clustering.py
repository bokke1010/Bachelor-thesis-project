# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# An adaptive clustering algorithm that considers all pairs of clusters.

from Adaptive_PCA.clustering import Cluster
from globals import kappa, clustering_threshold
import numpy as np

kapfac = 1 / kappa

def distance(A, B):
    """Adaptive distance function between two clusters.
    Distance is inflated if both clusters are already large."""
    base_distance = np.inner(A.center - B.center, A.center - B.center)
    return base_distance * kapfac if min(len(A.indices), len(B.indices)) > 200 else base_distance 


def clustering(clusters, threshold = clustering_threshold):
    """Compares the distance between every pair of clusters, merging
    pairs that fall below the threshold."""
    a, b = 0, 1
    high = len(clusters)
    if high < 2:
        return clusters
    while b != high:
        A, B = clusters[a], clusters[b]
        if distance(A, B) < threshold:
            A.merge(B)
            # Remove (b), clusters[high] will be out of bounds after truncation.
            high -= 1
            clusters[b], clusters[high] = clusters[high], clusters[b]
            if a == high:
                # Frustrating error case
                clusters[b], clusters[high-1] = clusters[high-1], clusters[b]
                a -= 1
            b = 0
        else:
            b += 1
            if b == high:
                a += 1
                b = a + 1
        if b == a:
            b += 1
    return clusters[:high]
