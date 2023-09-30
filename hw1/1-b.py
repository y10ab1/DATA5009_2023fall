import numpy as np
import pandas as pd
from LU import LU
from QR import QR


def power_iteration(X, n_iter=100):
    """
    Compute the largest eigenvalue and the corresponding eigenvector of a symmetric matrix X using power iteration.
    
    Parameters
    ----------
    X : ndarray
        The input matrix.
        
    Returns
    -------
    eigenvalue : float
        The largest eigenvalue of X.
    eigenvector : ndarray
        The corresponding eigenvector of the largest eigenvalue.
    """
    # initialize the eigenvector
    eigenvector = np.random.rand(X.shape[0])
    eigenvector = eigenvector / np.linalg.norm(eigenvector)
    
    # power iteration
    for _ in range(n_iter):
        eigenvector = np.dot(X, eigenvector)
        eigenvector = eigenvector / np.linalg.norm(eigenvector)
    
    # compute the eigenvalue
    eigenvalue = np.dot(np.dot(X, eigenvector), eigenvector) / np.dot(eigenvector, eigenvector)
    
    return eigenvalue, eigenvector
    

if __name__ == "__main__":
    # load X_picked.npy's data
    X_picked = np.load('X_picked.npy')
    
    # compute the inverse of X_picked
    X_inv = inverse(X_picked)
    print(X_inv)
    
    # validate the inverse of X_picked
    print(f'X_inv is the inverse of X_picked: {np.allclose(np.dot(X_picked, X_inv), np.eye(X_picked.shape[0]))}')
    