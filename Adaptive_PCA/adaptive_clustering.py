from vector import norm2_2
from Adaptive_PCA.clustering import Cluster
import numpy as np

kappa = 0.7
def distance(A, B):
    if len(A.indices) > 200 and len(B.indices) > 200:
        return norm2_2(A.center - B.center) / kappa
    else:
        return norm2_2(A.center - B.center)


def clustering(clusters, threshold = 100):
    a, b = 0, 1
    high = len(clusters)
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