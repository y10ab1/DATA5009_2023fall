import numpy as np
import pandas as pd
from QR import QR
    

if __name__ == "__main__":
    # load X_picked.npy's data
    X_picked = np.load('X_picked.npy')
    
    # compute all eigenvectors with REAL eigenvalues of X_picked using QR algorithm
    eigenvectors = []
    X_picked_copy = X_picked.copy()
    for _ in range(100):
        Q, R = QR(X_picked_copy)
        X_picked_copy = np.dot(R, Q)
        eigenvectors.append(X_picked_copy.diagonal())
    eigenvectors = np.array(eigenvectors).T
    print('The eigenvectors of X_picked are', eigenvectors)