import numpy as np
import pandas as pd
from LU import LU
    

if __name__ == "__main__":
    # load X_picked.npy's data
    X_picked = np.load('X_picked.npy')
    
    # compute LU decomposition of X_picked
    L, U = LU(X_picked)
    print(L)
    print(U)