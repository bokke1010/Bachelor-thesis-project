import pywt
import numpy as np


def remove_diagonal(image):
    (A, (H, V, D)) = pywt.dwt2(image, "haar")
    Dt = np.zeros_like(D)

    # 4th channel is identical to third channel, reason unknown
    return pywt.idwt2((A, (H, V, Dt)), "haar")[:,:,:3]