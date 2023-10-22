import pyforest
from SVD import SVD

def rank_k_approximation(X, k=None):
    """
    Find the rank-k approximation of X using SVD.
    :param X: matrix X
    :param k: number of principal components
    :return: rank-k approximation of X
    """
    k = X.shape[1] if k is None else k
    
    # compute the singular value decomposition of X
    U, S, V = SVD(X)
    # find the rank-k approximation of X
    X_k = np.dot(U[:, :k], np.dot(S[:k, :k], V[:k, :]))
    return X_k


if __name__ == "__main__":
    X_picked = np.load('X_picked.npy')
    print(X_picked)
    
    # rank-3 approximation of X_picked
    X_k = rank_k_approximation(X_picked, 3)
    print(X_k.round(2))
    
    