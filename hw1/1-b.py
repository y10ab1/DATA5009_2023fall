import numpy as np
import pandas as pd
from LU import LU
from QR import QR


def inverse(X):
    """
    Inverse of matrix X
    :param X: matrix X
    :return: X_inv
    """
    # check if X is a square matrix
    assert X.shape[0] == X.shape[1], 'X is not a square matrix, please try again'
    # check if X is invertible
    assert np.linalg.det(X) != 0, 'X is not invertible, please try again'
    # compute the inverse of X
    X_temp = np.hstack((X, np.eye(X.shape[0])))
    for i in range(X_temp.shape[0]):
        X_temp[i, :] = X_temp[i, :] / X_temp[i, i]
        for j in range(X_temp.shape[0]):
            if i != j:
                X_temp[j, :] = X_temp[j, :] - X_temp[i, :] * X_temp[j, i]
    X_inv = X_temp[:, X.shape[0]:]
    return X_inv
    

if __name__ == "__main__":
    # load X_picked.npy's data
    X_picked = np.load('X_picked.npy')
    
    # compute the inverse of X_picked
    X_inv = inverse(X_picked)
    print(X_inv)
    
    # validate the inverse of X_picked
    print(f'X_inv is the inverse of X_picked: {np.allclose(np.dot(X_picked, X_inv), np.eye(X_picked.shape[0]))}')
    