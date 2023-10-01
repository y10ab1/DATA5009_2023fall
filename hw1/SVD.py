import numpy as np

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

def SVD(X):
    """
    SVD decomposition/ SVD factorization
    :param X: matrix X
    :return: U, S, V
    """
    n = len(X)
    U = np.zeros((n, n))
    S = np.zeros((n, n))
    V = np.zeros((n, n))
    # compute eigenvalues and eigenvectors using power iteration
    for i in range(n):
        eigenvalue, eigenvector = power_iteration(X)
        # update U, S, V
        U[:, i] = eigenvector
        S[i, i] = eigenvalue
        V[:, i] = np.dot(X, eigenvector) / eigenvalue
        # update X
        X = X - eigenvalue * np.outer(eigenvector, eigenvector)
    return U, S, V
    

if __name__ == "__main__":
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 7]]).astype(float)
    # check X is full rank
    assert np.linalg.matrix_rank(X) == X.shape[0], 'X is not full rank, please try again'

    U, S, V = SVD(X)
    print(U)
    print(S)
    print(V)
    
    # validate the SVD decomposition of X
    print(f'U is orthogonal: {np.allclose(np.dot(U, U.T), np.eye(X.shape[0]))}')
    print(f'S is diagonal: {np.allclose(np.diag(np.diag(S)), S)}')
    print(f'V is orthogonal: {np.allclose(np.dot(V, V.T), np.eye(X.shape[0]))}')
    print(f'U * S * V.T = X: {np.allclose(np.dot(np.dot(U, S), V.T), X)}')