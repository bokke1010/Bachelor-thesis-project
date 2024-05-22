import numpy as np
from Wiener_filter import lpaici

sigma = 0

def autocovar(x, h, data):
    window = lpaici.construct_window(x, h, data)
    return np.sum(data[window]**2) / len(window)

def maxAttenuation(sigma, Rx, beta = 0.7):
    # Find attenuation in [0, 1] so that
    def costfunction(atten):
        sq = sigma**2
        g0 = sq / Rx
        hs = 1 - atten * g0
        gRg = g0**2 * Rx
        return (sq + gRg - 2 * sq * g0) / (sq + atten**2 * gRg - 2 * atten * sq * g0) - beta * atten**2

    space = np.linspace(0,1,100)
    # Assume costfunction is automatically vectorizeable
    max_index = np.argmax(costfunction(space))

    return space[max_index]

def wiener_filter(data):
    filtered_data = np.empty_like(data)
    for x in range(len(data)):
        bandwidth = lpaici.Intersection_of_Intervals(x, data)
        Rx = autocovar(x, bandwidth, data)

        attenuation = maxAttenuation(sigma, Rx)
        # v should be a filter
        hs = 1 - attenuation * sigma * sigma / Rx

        filtered_data[x] = hs * data[x]
    return filtered_data
