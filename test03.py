import numpy as np
from tools.np_imageload import load_image, save_image
from Remove_diagonal.wavelet import remove_diagonal
from Visualize.show_image import show_image

denoised_ZM = load_image("noiseZM.png") - 128

changed = np.asarray( np.clip(remove_diagonal(denoised_ZM) + 128,0,255), dtype="uint8")
print(np.std((changed - denoised_ZM).flatten()))
print(np.std((denoised_ZM).flatten()))
print(np.std((changed).flatten()))