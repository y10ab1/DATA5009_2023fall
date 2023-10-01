# QR decomposition/ QR factorization
import numpy as np

def QR(X):
    """
    QR decomposition/ QR factorization
    :param X: matrix X
    :return: Q, R
    """
    X = X.copy().astype(float)
    n = len(X)
    Q = np.eye(n)
    R = np.zeros((n, n))
    Q[:, 0] = X[:, 0] / np.linalg.norm(X[:, 0])
    
    for i in range(1, n):
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], X[:, i])
            X[:, i] -= R[j, i] * Q[:, j]
        print(X[:, i], np.linalg.norm(X[:, i]))
        Q[:, i] = X[:, i] / np.linalg.norm(X[:, i])
    for i in range(n):
        R[i, i] = np.dot(Q[:, i], X[:, i])
    return Q, R

if __name__ == "__main__":
    X = np.load('X_picked.npy')
    # ensure X is can be decomposed
    assert np.linalg.matrix_rank(X) == X.shape[0], 'X cannot be decomposed due to rank deficiency'
    
    
    Q, R = QR(X)
    print(Q)
    print(R)
    
    # validate the QR decomposition of X
    print(f'Q is orthogonal: {np.allclose(np.dot(Q, Q.T), np.eye(X.shape[0]))}')
    print(f'R is upper triangular: {np.allclose(np.triu(R), R)}')
    print(f'Q * R = X: {np.allclose(np.dot(Q, R), X)}')