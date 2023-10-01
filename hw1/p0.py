import numpy as np
import pandas as pd

def random_pick(X, n=6):
    """
    Randomly pick n rows and n cols from X
    :param X: matrix X
    :param n: number of rows and cols to pick
    :return: X_picked
    """
    # randomly pick n rows and n cols from X (default n=6 and uniform sampling without replacement)
    idxs_r = np.random.choice(X.shape[0], n, replace=False)
    X_picked = X[idxs_r, :]
    idxs_c = np.random.choice(X.shape[1], n, replace=False)
    X_picked = X_picked[:, idxs_c]
    
    # print where the rows and cols are picked from
    print(f'Rows are picked from {idxs_r}')
    print(f'Cols are picked from {idxs_c}')
    
    # check if X_picked has at least 4 eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(X_picked)
    print(f'X_picked has {len(eigenvalues)} eigenvalues')
    assert len(eigenvalues) >= 4, 'X_picked does not have at least 4 eigenvalues, please try again'
    
    return X_picked
    
    
    
if __name__ == "__main__":
    # load CAmaxTemp.csv's data
    df = pd.read_csv('CAmaxTemp.csv')
    # keep only the temperature data
    X = df.iloc[:, 4:-1].values
    # randomly pick 6 rows and 6 cols from X
    X_picked = random_pick(X)
    # print the picked data
    print(X_picked)
    
    # save the picked data to a npy file
    np.save('X_picked.npy', X_picked)
    np.save('X.npy', X)
    
    