import numpy as np
import matplotlib.pyplot as plt

def show_fingerprint(fingerprint):
    if isinstance(fingerprint, str):
        with open(fingerprint, 'rb') as f:
            fingerprint = np.load(f)

    plt.imshow(fingerprint + 0.5,origin='lower')
    plt.show()

if __name__ == "__main__":
    import sys
    show_fingerprint(sys.argv[1])
