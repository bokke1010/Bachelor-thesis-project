import numpy as np

def circular_cross_correlation_1d(k, a, b):
    """Determine the circular cross-correlation between
    two vectors in R^n"""
    n, nc = len(a), len(b)
    assert n == nc
    return np.inner(a, np.roll(b, k)) / n


def circular_cross_correlation_2d(W, R, x, y):
    """Determine the circular cross-correlation between
    two vectors in R^n"""
    n = W.shape
    return np.sum(np.square(W, np.roll(R, (x, y)))) / np.prod(n)

def peak_correlation_energy(W, R, omega):
    n, nc = W.shape, R.shape
    assert n == nc
    top = circular_cross_correlation_2d(W,R,0)
    total = 0
    for y in range(n[0]):
        for x in range(n[1]):
            if (x,y) not in omega:
                total += circular_cross_correlation_2d(W,R,x,y)
    bottom = total / (np.prod(n) - len(omega))
    return top * top / bottom

def signed_peak_correlation_energy(W, R, omega):
    n, nc = W.shape, R.shape
    assert n == nc
    top = circular_cross_correlation_2d(W,R,0)
    total = 0
    for y in range(n[0]):
        for x in range(n[1]):
            if (x,y) not in omega:
                total += circular_cross_correlation_2d(W,R,x,y)
    bottom = total / (np.prod(n) - len(omega))
    return np.abs(top) * top / bottom
