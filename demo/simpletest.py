# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# A small demonstration of the denoiser.
# Needs to be moved to the project root to run as it
# requires a main import.

import main
import matplotlib.pyplot as plt

cat = main.load_image("muis_small.png")

main.large_window_size = 256

image = main.denoise((0, cat[:,:,0]))

plt.imshow(image / 255.0)
plt.show()

