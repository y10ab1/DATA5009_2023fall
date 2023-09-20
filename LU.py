# LU decomposition/ LU factorization

import numpy as np

def LU(X):
    """
    LU decomposition/ LU factorization
    :param X: matrix X
    :return: L, U
    """
    n = len(X)
    L = np.eye(n) # np.eye(n) return a 2-D array with 1s on the diagonal and 0s elsewhere.
    U = np.zeros((n, n)) # np.zeros((n, n)) return a 2-D array with 0s on the diagonal and 0s elsewhere.
    for i in range(n):
        for j in range(i, n):
            U[i, j] = X[i, j] - np.dot(L[i, :i], U[:i, j])
        for j in range(i + 1, n):
            L[j, i] = (X[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]
    return L, U

def LU_PP(X):
    """
    LU decomposition/ LU factorization with partial pivoting
    :param X: matrix X
    :return: L, U
    """
    n = len(X)
    L = np.eye(n)
    U = np.zeros((n, n))
    P = np.eye(n)
    for i in range(n):
        # find the pivot
        pivot = np.argmax(np.abs(X[i:, i])) + i
        # swap rows
        X[[i, pivot], :] = X[[pivot, i], :]
        P[[i, pivot], :] = P[[pivot, i], :]
        for j in range(i, n):
            U[i, j] = X[i, j] - np.dot(L[i, :i], U[:i, j])
        for j in range(i + 1, n):
            L[j, i] = (X[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]
    return L, U, P
    
def LU_FP(X):
    """
    LU decomposition/ LU factorization with full pivoting
    :param X: matrix X
    :return: L, U
    """
    n = len(X)
    L = np.eye(n)
    U = np.zeros((n, n))
    P = np.eye(n)
    Q = np.eye(n)
    for i in range(n):
        # find the pivot
        pivot = np.argmax(np.abs(X[i:, i])) + i
        # swap rows
        X[[i, pivot], :] = X[[pivot, i], :]
        P[[i, pivot], :] = P[[pivot, i], :]
        X[:, [i, pivot]] = X[:, [pivot, i]]
        Q[:, [i, pivot]] = Q[:, [pivot, i]]
        for j in range(i, n):
            U[i, j] = X[i, j] - np.dot(L[i, :i], U[:i, j])
        for j in range(i + 1, n):
            L[j, i] = (X[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]
    return L, U, P, Q

def LDU(X):
    """
    LDU decomposition/ LDU factorization
    :param X: matrix X
    :return: L, D, U
    """
    n = len(X)
    L = np.eye(n)
    D = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            U[i, j] = X[i, j] - np.dot(L[i, :i], D[:i, :i]).dot(U[:i, j])
        for j in range(i, n):
            D[i, i] = X[i, i] - np.dot(L[i, :i], D[:i, :i]).dot(U[:i, i])
        for j in range(i + 1, n):
            L[j, i] = (X[j, i] - np.dot(L[j, :i], D[:i, :i]).dot(U[:i, i])) / D[i, i]
    return L, D, U


if __name__ == "__main__":
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    L, U = LU(X)
    print(L)
    print(U)