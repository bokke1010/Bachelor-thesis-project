# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# Some preparations for an eventual RISS implementation.

import numpy as np
from Visualize.show_image import show_image

def RISS(image):
    """Dummy implementation, does nothing useful."""
    print(np.std(np.real(image)))
    freqdom = np.fft.fft2(image, axes=(0,1))
    
    re_signal, im_signal = np.real(freqdom), np.imag(freqdom)


    recombined = np.empty_like(image, dtype=np.complex128)
    recombined.real = re_signal
    recombined.imag = im_signal
    reversed_img = np.fft.ifft2(recombined, axes=(0,1))
    print(np.std(np.real(reversed_img)))
    show_image(128+np.real(reversed_img))