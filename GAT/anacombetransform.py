# The image can be described as 
# I = η P(I_0) + ψ
# Where I is the image, I_0 is the noise-free image,
# P(I_0) is the poisson distribution of I_0, η is the
# poisson gain and ψ is the gaussian noise with mean μ
# and standard deviation σ

import numpy as np

gain = 1
scale = 0.375
stdev = 0

def anacombe(I):
    im = gain * I + scale * gain * gain + stdev
    return 2.0 * np.sqrt(np.maximum(0.0, im)) / gain

def inv_anacombe(Ivd):
    return 0.25 * Ivd * Ivd + (0.25 - (1.375 + 0.625 * np.sqrt(3.0/2.0) / Ivd) / Ivd) / Ivd - 0.125 - stdev * stdev