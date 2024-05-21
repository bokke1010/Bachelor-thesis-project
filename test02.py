import numpy as np
from Eigenvector.TriDiag import TriDiagonalize


A = np.ones((3,3))
# A = np.random.rand(5,5)
TriDiagonalize(A)
print(A)