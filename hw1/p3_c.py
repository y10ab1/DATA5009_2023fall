import pyforest

def centering(x):
    return x - np.mean(x)

def visualize_for_each_step(X):
    """
    Visualize the data for each step, and plot each step's data in a figure in 3D.
    :param X: matrix X
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
    print(X_decorrelated.round(2))
    
    # compute the whitened matrix X
    X_whitened = np.apply_along_axis(lambda x: x / np.sqrt(eigenvalues), 1, X_decorrelated)
    print(X_whitened.round(2))
    
    # plot the data for each step in a figure with 4 subplots in 3D
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2])
    ax1.set_title('X')
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(X_centered[:, 0], X_centered[:, 1], X_centered[:, 2])
    ax2.set_title('X_centered')
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(X_decorrelated[:, 0], X_decorrelated[:, 1], X_decorrelated[:, 2])
    ax3.set_title('X_decorrelated')
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(X_whitened[:, 0], X_whitened[:, 1], X_whitened[:, 2])
    ax4.set_title('X_whitened')
    plt.tight_layout()
    plt.savefig('p3_c_visualize_for_each_step.png')
    

if __name__ == '__main__':
    X = np.load('X.npy')
    # X = np.array([[1, 1, 2, 0, 5, 4, 5, 3],[3, 2, 3, 3, 4, 5, 5, 4]]).astype(float).T
    # choose 1, 5 and 10 cols of X
    X_prime = X[:,[1, 5, 9]]
    print(X_prime)
    print(X)
    
    visualize_for_each_step(X_prime)