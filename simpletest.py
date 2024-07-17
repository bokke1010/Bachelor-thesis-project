# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# A small demonstration of the denoiser.
# Needs to be moved to the project root to run as it
# requires a main import.

import main
import matplotlib.pyplot as plt
from tools.np_imageload import load_image

cat = load_image("muis_small.png")

for i in range(3):
    cat[:,:,i] = main.denoise(cat[:,:,i])

plt.imshow(cat / 255.0)
plt.show()
