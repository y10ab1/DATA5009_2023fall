import pyforest
import numpy as np

def centering(x):
    return x - np.mean(x)

def var_covar_matrix(X):
    """
    Compute the variance-covariance matrix of X.
    :param X: matrix X
    :return: var_covar_matrix
    """
    # centering X along each row
    X_centered = np.apply_along_axis(centering, 1, X)
    # print(X_centered.round(2))
    
    # compute the variance-covariance matrix of X
    var_covar_matrix = np.dot(X_centered, X_centered.T) / (X.shape[1] - 1)
    return var_covar_matrix

if __name__ == '__main__':
    X_picked = np.load('X_picked.npy')
    print(X_picked)
    
    # compute the variance-covariance matrix of X_picked and print it
    var_covar_matrix = var_covar_matrix(X_picked)
    print(var_covar_matrix.round(2))

    # validate the result using np.cov
    print(f'var_covar_matrix is correct: {np.allclose(var_covar_matrix, np.cov(X_picked))}')
    