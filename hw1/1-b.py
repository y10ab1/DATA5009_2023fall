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
    
    # compute the largest eigenvalue and the corresponding eigenvector of X_picked using power iteration
    eigenvalue, eigenvector = power_iteration(X_picked)
    print('The largest eigenvalue of X_picked is', eigenvalue)
    print('The corresponding eigenvector of the largest eigenvalue is', eigenvector)
    
    # validate the result using numpy.linalg.eig
    print('Computed by numpy.linalg.eig:')
    print('The largest eigenvalue of X_picked is', np.linalg.eig(X_picked)[0][0])
    print('The corresponding eigenvector of the largest eigenvalue is', np.linalg.eig(X_picked)[1][:, 0])