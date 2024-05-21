import numpy as np

def norm2_2(v):
    return np.sum((v)**2)
def dist2(a, b):
    return norm2_2(a-b)
