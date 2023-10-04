import numpy as np
from QR import QR

if __name__ == "__main__":
    # load X_picked.npy's data
    X_picked = np.load('X_picked.npy')
    
    # compute all eigenvectors with REAL eigenvalues of X_picked using QR algorithm
    eigenvectors = []
    eigenvalues = []
    X_picked_copy = X_picked.copy()
    tol = 16
    
    # the QR algorithm, iteration stops when all the off-diagonal elements are less than tol
    while np.abs(X_picked_copy - np.diag(np.diag(X_picked_copy)))[0, 1] > tol:
        Q, R = QR(X_picked_copy)
        X_picked_copy = np.dot(R, Q)
        eigenvectors.append(Q)
        eigenvalues.append(np.diag(X_picked_copy))
        print(f"off-diagonal elements: {np.abs(X_picked_copy - np.diag(np.diag(X_picked_copy)))[0, 1]}")
    print('The eigenvectors of X_picked are:')
    print(np.array(eigenvectors)[-1].round(2))
    print('The eigenvalues of X_picked are:')
    print(np.array(eigenvalues)[-1].round(2))
    
    print()
    print(np.linalg.eig(X_picked)[1].round(2))
    print(np.linalg.eig(X_picked)[0].round(2))