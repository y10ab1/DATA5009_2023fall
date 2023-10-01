import pyforest
from SVD import SVD

def find_PC_using_SVD(X, k=None):
    """
    Find the first k principal components of X using SVD.
    :param X: matrix X
    :param k: number of principal components
    :return: first k principal components of X
    """
    k = X.shape[1] if k is None else k
    
    # compute the singular value decomposition of X
    U, S, V = SVD(X)
    # find the first k principal components of X
    PC = U[:, :k]
    return PC


if __name__ == "__main__":
    X_picked = np.load('X_picked.npy')
    print(X_picked)
    
