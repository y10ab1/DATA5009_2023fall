import numpy as np
import pandas as pd

# set the seed
np.random.seed(825)

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
    

    
    # check if X_picked has at least 4 eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(X_picked)
    print(f'X_picked has {len(eigenvalues)} eigenvalues')
    assert len(eigenvalues) >= 4, 'X_picked does not have at least 4 eigenvalues, please try again'
    
    return X_picked, idxs_r, idxs_c
    
    
    
if __name__ == "__main__":
    # load CAmaxTemp.csv's data
    df = pd.read_csv('CAmaxTemp.csv')
    # keep only the temperature data
    X = df.iloc[:, 4:-1].values
    print(df)
    # randomly pick 6 rows and 6 cols from X
    X_picked, idxs_r, idxs_c = random_pick(X)
    # print where the stations and months are picked from
    print(f'Stations are picked from:\n {df.iloc[idxs_r, 0].values}\n')
    print(f'Months are picked from:\n {df.columns[4:-1][idxs_c].values}\n')
    # print the picked data
    print(X_picked)
    # save the picked data to a npy file
    np.save('X_picked.npy', X_picked)
    np.save('X.npy', X)
    
    