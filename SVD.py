import numpy as np

def SVD(X):
    """
    SVD decomposition/ SVD factorization
    :param X: matrix X
    :return: U, S, V
    """
    n = len(X)
    U = np.eye(n)
    S = np.zeros((n, n))
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            S[i, j] = X[i, j] - np.dot(U[i, :i], np.dot(S[:i, :i], V[:i, j]))
        for j in range(i + 1, n):
            U[j, i] = (X[j, i] - np.dot(U[j, :i], np.dot(S[:i, :i], V[:i, i]))) / S[i, i]
    return U, S, V

if __name__ == "__main__":
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    X = np.array([[1, 0, 0, 0, 2],
                  [0, 0, 3, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 2, 0 ,0 ,0] ])
    U, S, V = SVD(X)
    print(U)
    print(S)
    print(V)