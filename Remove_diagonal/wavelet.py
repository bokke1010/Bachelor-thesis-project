# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# This file contains only the remove diagonal high-frequency noise step
# This step uses the 2D digital wavelet transform in order to remove
# a significant amount of very high-frequency diagonal noise, and improve
# PRNU image matching.

import pywt
import numpy as np

def remove_diagonal(image):
    """Removes the highest frequency diagonal noise component."""
    (A, (H, V, D)) = pywt.dwt2(image, "haar")
    Dt = np.zeros_like(D)

    # 4th channel is identical to third channel, reason unknown
    return pywt.idwt2((A, (H, V, Dt)), "haar")[:,:,:3]