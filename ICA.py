import numpy as np

def ICA(X, n_components=2, max_iter=1000, tol=1e-5):
    """
    ICA decomposition/ ICA factorization
    :param X: matrix X
    :return: W, S
    """
    n = len(X)
    W = np.random.rand(n_components, n)
    for i in range(n_components):
        W[i, :] = W[i, :] / np.linalg.norm(W[i, :])
    for i in range(max_iter):
        W_old = W.copy()
        for j in range(n_components):
            W[j, :] = np.mean(X * g(np.dot(W, X)), axis=1) - np.mean(g_prime(np.dot(W, X))) * W[j, :]
            W[j, :] = W[j, :] / np.linalg.norm(W[j, :])
        if np.linalg.norm(W - W_old) < tol:
            break
    S = np.dot(W, X)
    return W, S

def g(x):
    return np.tanh(x)

def g_prime(x):
    return 1 - np.tanh(x) ** 2

if __name__ == "__main__":
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    W, S = ICA(X)
    print(W)
    print(S)