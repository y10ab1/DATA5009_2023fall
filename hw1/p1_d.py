import numpy as np
from QR import QR

def inverse_using_QR(X):
    """
    Caculate inverse of matrix X using Q and R.
    :param X: matrix X
    :return: X_inv
    """
    # since Q is orthogonal, Q.T is the inverse of Q
    Q, R = QR(X)
    Q_T = Q.T
    # since R is upper triangular, we can use back substitution to solve X_inv
    X_inv = np.zeros(X.shape)
    for i in range(X.shape[0] - 1, -1, -1):
        for j in range(X.shape[1] - 1, i, -1):
            X_inv[i, :] -= X_inv[j, :] * R[i, j]
        X_inv[i, :] += Q_T[i, :]
        X_inv[i, :] /= R[i, i]
    return X_inv
    

if __name__ == "__main__":
    # load X_picked.npy's data
    X_picked = np.load('X_picked.npy')
    
    # compute the inverse of X_picked
    X_inv = inverse_using_QR(X_picked)
    print(X_inv.round(2))
    
    # validate the inverse of X_picked
    print(f'X_inv is the inverse of X_picked: {np.allclose(np.dot(X_picked, X_inv), np.eye(X_picked.shape[0]))}')
    print(np.linalg.inv(X_picked))
    