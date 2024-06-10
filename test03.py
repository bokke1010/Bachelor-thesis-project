import numpy as np
from tools.np_imageload import load_image, save_image
from Visualize.show_image import show_image
from RISS.smoothing import RISS

denoised_ZM = load_image("noiseZM.png") - 128
RISS(denoised_ZM)
