# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# A few helper functions to easily load and save images
# into and from numpy arrays.

import numpy as np
from PIL import Image

def load_image( infilename ) :
    """Creates a numpy int32 array containing the image at infilename."""
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    """Saves a (n, m, 3)-shaped numpy array into an image at outfilename, clipping the data to the range 0 - 255."""
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "RGB" )
    img.save( outfilename )

def save_image_grayscale( npdata, outfilename ):
    """Saves a (n, m)-shaped numpy array into a grayscale image at outfilename, clipping the data to the range 0 - 255."""
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )