import numpy as np

def crosscorr(W, R):
    sW, sR = W - np.mean(W), R - np.mean(R)
    return np.dot(sW, sR) / (np.linalg.norm(sW) * np.linalg.norm(sR))
