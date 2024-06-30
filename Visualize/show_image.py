# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# A very simple function that clips image data and shows it.

import matplotlib.pyplot as plt
import numpy as np

def show_image(image):
    """Show an image with 1 or 3 channels. Data range [0-255]."""
    plt.imshow(np.asarray( np.clip(image,0,255), dtype="uint8"))
    plt.show()