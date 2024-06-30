# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# Contains a number of functions to utilize
# the suboptimal Wiener filter in the context of my thesis.


import numpy as np
from Wiener_filter import lpa_ici

sigma = 0

def auto_covariance(x, h, data):
    window = lpa_ici.construct_window(x, h, data)
    return np.sum(data[window]**2) / len(window)

def max_attenuation(sigma, Rx, beta = 0.7):
    """Determine which attenuation value (within (0,1)) maximizes the cost function."""
    # Find attenuation in [0, 1] so that
    def costfunction(atten):
        sq = sigma**2
        g0 = sq / Rx
        # hs = 1 - atten * g0
        gRg = g0**2 * Rx
        return (sq + gRg - 2 * sq * g0) / (sq + atten**2 * gRg - 2 * atten * sq * g0) - beta * atten**2

    # This algorithm isn't great, but it'll work
    # It also isn't a bottleneck.
    rough_space = np.linspace(0,1,50)
    max_index = np.argmax(costfunction(rough_space))
    fine_space = np.linspace(rough_space[max(0, max_index - 1)], rough_space[min(1, max_index + 1)], 50)
    max_fine_index = np.argmax(costfunction(rough_space))

    return fine_space[max_fine_index]

def wiener_filter(data):
    """Apply the suboptimal Wiener filter, determining the bandwidth
    using the LPA-ICI technique and the attenuation using the cost function."""
    filtered_data = np.empty_like(data)
    for x in range(len(data)):
        bandwidth = lpa_ici.intersection_of_intervals(x, data)
        Rx = auto_covariance(x, bandwidth, data)

        attenuation = max_attenuation(sigma, Rx)
        hs = 1 - attenuation * sigma * sigma / Rx

        filtered_data[x] = hs * data[x]
    return filtered_data
