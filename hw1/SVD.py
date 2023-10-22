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
    n = len(X)
    U = np.zeros((n, n))
    S = np.zeros((n, n))
    Vt = np.zeros((n, n))
    
    # Consider X^T @ X for power iteration
    A = np.dot(X.T, X)
    
    for i in range(n):
        eigenvalue, eigenvector = power_iteration(A)
        
        
        # Singular value is square root of eigenvalue
        singular_value = np.sqrt(eigenvalue)
        S[i, i] = singular_value
        
        # V's column is the eigenvector
        Vt[i, :] = eigenvector
        
        # U's column is normalized version of X*v/s
        u = np.dot(X, eigenvector) / singular_value
        U[:, i] = u
        
        # Rank-one update to A
        A = A - eigenvalue * np.outer(eigenvector, eigenvector)
        V = Vt.T
    return U, S, V

def SVD_(A):
    # Compute eigendecomposition of A^T A to get V and singular values
    eigenvalues, V = np.linalg.eig(np.dot(A.T, A))
    singular_values = np.sqrt(eigenvalues)
    # Sort singular values in descending order
    idx = np.argsort(singular_values)[::-1]
    singular_values = singular_values[idx]
    V = V[:, idx]
    # Compute U
    U = np.zeros(A.shape)
    for i in range(A.shape[1]):
        U[:, i] = np.dot(A, V[:, i]) / singular_values[i]
    # Construct S
    S = np.diag(singular_values)
    
    # nan to 0
    U = np.nan_to_num(U)
    S = np.nan_to_num(S)
    V = np.nan_to_num(V)
    
    return U, S, V
    

    

if __name__ == "__main__":
    # X = np.array([[1, 0, 0, 0, 2], [0, 0, 3, 0, 0], [0, 0, 0, 0, 0], [0, 2, 0, 0, 0]]).astype(float)
    X = np.load('X_picked.npy')
    # check X is full rank
    # assert np.linalg.matrix_rank(X) == X.shape[0], 'X is not full rank, please try again'

    U, S, V = SVD_(X)
    print(U)
    print(S)
    print(V)
    
    # validate the SVD decomposition of X
    print(f'U is orthogonal: {np.allclose(np.dot(U, U.T), np.eye(U.shape[0]))}')
    print(f'S is diagonal: {np.allclose(np.diag(np.diag(S)), S)}')
    print(f'V is orthogonal: {np.allclose(np.dot(V, V.T), np.eye(V.shape[0]))}')
    print(f'U * S * Vt = X: {np.allclose(np.dot(np.dot(U, S), V.T), X)}')

