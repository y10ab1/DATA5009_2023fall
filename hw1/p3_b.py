import pyforest

def centering(x):
    return x - np.mean(x)

def decorrelation(X):
    """
    Decorrelate X.
    :param X: matrix X
    :return: decorrelated matrix X
    """
    # centering X along each row
    X_centered = np.apply_along_axis(centering, 0, X)
    print(X_centered.round(3))
    
    # compute the variance-covariance matrix of X
    var_covar_matrix = np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)
    print("var_covar_matrix")
    print(var_covar_matrix.round(2))
    
    # compute the eigenvalues and eigenvectors of the variance-covariance matrix of X
    eigenvalues, eigenvectors = np.linalg.eig(var_covar_matrix)
    print(eigenvalues.round(2))
    print(eigenvectors.round(2))
    
    # compute the decorrelated matrix X
    X_decorrelated = np.dot(X_centered, eigenvectors)
    return X_decorrelated, eigenvalues, eigenvectors

def whitening(X):
    """
    Whitening X.
    :param X: matrix X
    :return: whitened matrix X
    """
    # compute the decorrelated matrix X
    X_decorrelated, eigenvalues, eigenvectors = decorrelation(X)
    print(X_decorrelated.round(2))
    
    # compute the whitened matrix X
    X_whitened = np.apply_along_axis(lambda x: x / np.sqrt(eigenvalues), 1, X_decorrelated)
    
    return X_whitened

if __name__ == '__main__':
    X = np.load('X.npy')
    # X = np.array([[1, 1, 2, 0, 5, 4, 5, 3],[3, 2, 3, 3, 4, 5, 5, 4]]).astype(float).T
    # choose 1, 5 and 10 cols of X
    X_prime = X[:,[1, 5, 9]]
    print(X_prime)
    print(X)
    
    print('Whitening X_prime')
    X_prime_whitened = whitening(X_prime)
    print(X_prime_whitened.round(2))