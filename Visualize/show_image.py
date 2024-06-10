import matplotlib.pyplot as plt
import numpy as np

def show_image(image):
    plt.imshow(np.asarray( np.clip(image,0,255), dtype="uint8"))
    plt.show()