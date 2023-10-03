import pyforest

def fastICA(X, n_components=None, max_iter=200, tol=1e-4):
    """
    Perform FastICA algorithm to find the independent components of X.
    :param X: matrix X (n_samples, n_components)
    :param n_components: number of independent components
    :param max_iter: maximum number of iterations
    :param tol: tolerance
    :return: independent components of X
    """
    n_components = X.shape[1] if n_components is None else n_components
    n_samples = X.shape[0]
    
    # initialize the weight matrix W
    W = np.random.rand(n_components, n_components)
    # normalize the weight matrix W
    W = np.dot(np.linalg.inv(np.dot(W, W.T)), W)
    
    # initialize the independent components of X
    IC = np.zeros((n_samples, n_components))
    
    # perform FastICA algorithm
    for i in range(n_components):
        for _ in range(max_iter):
            # compute the independent components of X
            IC[:, i] = np.dot(X, W[i, :])
            # compute the first derivative of g
            g_prime = np.tanh(IC[:, i])
            # compute the second derivative of g
            g_prime_prime = 1 - np.power(g_prime, 2)
            # update the weight matrix W
            W[i, :] = np.mean(X * g_prime.reshape(-1, 1), axis=0) - np.mean(g_prime_prime) * W[i, :]
            # normalize the weight matrix W
            W = np.dot(np.linalg.inv(np.dot(W, W.T)), W)
            
            # check convergence
            if np.allclose(np.mean(g_prime_prime) * W[i, :], 0, atol=tol):
                break
    return IC

if __name__ == '__main__':
    X = np.load('X.npy')
    # choose 1, 5 and 10 cols of X
    X_prime = X[:,[1, 5, 9]]
    print(X)
    print(X_prime)
    
    # find 3 independent components of X_prime
    IC = fastICA(X_prime, n_components=3)
    print(IC.round(2))
    
    # plot 3 independent components of X_prime
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(IC[:, 0])
    plt.subplot(1, 3, 2)
    plt.plot(IC[:, 1])
    plt.subplot(1, 3, 3)
    plt.plot(IC[:, 2])
    plt.savefig('p3_d.png')
    