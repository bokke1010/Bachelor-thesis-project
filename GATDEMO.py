# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# Contains a simple demo for a gaussian filter though
# the generalized Anscombe transform

import GAT.anscombetransform

from PIL import Image
import numpy as np
from scipy import ndimage

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )

sigma = 7

image = load_image("muis1.jpg")
red_channel = image[:,:,0] + np.random.normal(0, 50, image.shape[0:2])

GAT.anscombetransform.sigma = sigma
Image_v = GAT.anscombetransform.anscombe(red_channel)

Image_vd = ndimage.gaussian_filter(Image_v, sigma)

W = GAT.anscombetransform.inv_anscombe(Image_vd)

Image_naive = ndimage.gaussian_filter(red_channel, sigma)

save_image(Image_naive, "naivefiltered.jpg")
save_image(W, "GATfiltered.jpg")