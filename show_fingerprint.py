# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# A small helper function to visualize fingerprints and
# extracted PRNU noise.

import numpy as np
import matplotlib.pyplot as plt

def show_fingerprint(fingerprint_path):
    """Shows the fingerprint or PRNU noise saved at the given path"""
    fingerprint = None
    if isinstance(fingerprint_path, str):
        with open(fingerprint_path, 'rb') as f:
            fingerprint = np.load(f)

    plt.imshow(fingerprint + 0.5,origin='lower')
    plt.show()

if __name__ == "__main__":
    import sys
    show_fingerprint(sys.argv[1])
