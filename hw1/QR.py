# QR decomposition/ QR factorization
import numpy as np

def QR(X):
    """
    QR decomposition/ QR factorization
    :param X: matrix X
    :return: Q, R
    """
    n = len(X)
    Q = np.eye(n)
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            R[i, j] = X[i, j] - np.dot(Q[i, :i], R[:i, j])
        for j in range(i + 1, n):
            Q[j, i] = (X[j, i] - np.dot(Q[j, :i], R[:i, i])) / R[i, i]
    return Q, R

if __name__ == "__main__":
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    Q, R = QR(X)
    print(Q)
    print(R)
    
    # validate the QR decomposition of X
    print(f'Q is orthogonal: {np.allclose(np.dot(Q, Q.T), np.eye(X.shape[0]))}')
    print(f'R is upper triangular: {np.allclose(np.triu(R), R)}')
    print(f'Q * R = X: {np.allclose(np.dot(Q, R), X)}')