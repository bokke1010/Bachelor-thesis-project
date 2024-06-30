# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# A simple demo script to demonstrate the 2d discrete wavelet transform.

import numpy as np
import pywt

from PIL import Image

import matplotlib.pyplot as plt

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

wavelets = ["db3", "db7", "haar"]
wavelet = "db2"
reconstruct = False

img_blue = load_image("muis_small.png")[:,:,2]
(cA, (cH, cV, cD)) = pywt.dwt2(img_blue, wavelet)
plt.suptitle(f"Discrete 2D wavelet transform of a cat picture\nusing the {wavelet} wavelet.")

if reconstruct:
    cDm = np.zeros_like(cD)
    reconstructed = pywt.idwt2((cA, (cH, cV, cDm)), wavelet)
    imin, imax = np.min(reconstructed), np.max(reconstructed)
    plt.imshow((reconstructed + imin) / (imax + imin), cmap="gray")
else:
    for i, c in enumerate([cA, cH, cV, cD]):
        plt.subplot(2,2,i+1)
        imin, imax = np.min(c), np.max(c)
        plt.imshow((c + imin) / (imax + imin), cmap="gray")
plt.show()
