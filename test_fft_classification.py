import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fft2, ifft2, ifftshift
import sys
import os

def cross_correlate_2d(x, h):
    h = ifftshift(ifftshift(h, axes=0), axes=1)
    return ifft2(fft2(x) * np.conj(fft2(h))).real

# load image as greyscale

fingerprint = None
with open("fingerprinti8p.npy", 'rb') as f:
    fingerprint = np.load(f)

residues, names = [], []
with os.scandir("dataiotherFF") as it:
            for entry in it:
                if entry.is_file and entry.name.endswith(".npy"):
                    with open(entry.path, "rb") as f:
                        residues.append(np.load(f))
                        names.append(entry.name)

# make kernels

for name, residue in zip(names, residues):
    # plt.title(name)
    # plt.imshow((residue + 0.5) / 2.0); plt.xticks([]); plt.yticks([]); plt.show()
    # plt.clf()
    # compute
    out1 = cross_correlate_2d(residue[:,:,0], fingerprint[:,:,0])
    out2 = cross_correlate_2d(residue[:,:,1], fingerprint[:,:,1])
    out3 = cross_correlate_2d(residue[:,:,2], fingerprint[:,:,2])
    finccc = (out1 + out2 + out3) / 3
    plt.title(name)
    plt.imshow(finccc, cmap="gray"); plt.xticks([]); plt.yticks([]); plt.show()
    plt.clf()
    # continue
    total = 0
    rem = 0
    peak_size = 7
    n = residue.shape
    my, mx = 0, 0
    # my, mx = n[0]//2, n[1]//2
    top = finccc[my, mx]
    for y in range(n[0]):
        for x in range(n[1]):
            xd, yd = min(x, n[1]-x), min(y, n[0] - y)
            if (xd)**2 + (yd)**2 > peak_size*peak_size:
                total += finccc[y, x]
            else:
                rem += 1
    bottom = total / (np.prod(n) - rem)
    print(name, np.abs(top) * top / bottom)
    # # plot

