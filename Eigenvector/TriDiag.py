import numpy as np
from vector import norm2_2

def TriDiagonalize(A: np.array) -> np.array:
    (m, n) = A.shape
    for j in range(1, m):
        print(A)
        a = A[j:, j, np.newaxis]
        e1 = np.eye(m-j, 1)

        # Stable choice
        sig = np.sign(np.transpose(e1) @ a)
        if sig == 0:
            sig = -1
        
        u = a + sig * e1 * np.sqrt(norm2_2(a))
        print("a: ", a, "\nu: ", u)
        u = u / np.sqrt(norm2_2(u))

        # H = I - 2 u u' I ; householder matrix
        # Calculate H A H^T efficiently:
        A[j:m,:] = A[j:m,:] - 2 * u @ (np.transpose(u) @ A[j:m,:])
        A[:,j:n] = A[:,j:n] - 2 * (A[:,j:n] @ u) @ np.transpose(u)
        avg = (A[j-1,j] + A[j,j-1]) / 2
        A[j-1,j] = avg
        A[j,j-1] = avg
        A[j+1:m,j-1] = 0
        A[j-1,j+1:n] = 0
