import pyforest

def kurtosis(x):
    """
    Compute the kurtosis of x.
    :param x: vector x
    :return: kurtosis of x
    """
    # compute the mean of x
    mean = np.mean(x)
    # compute the standard deviation of x
    std = np.std(x)
    # compute the kurtosis of x
    kurtosis = np.mean((x - mean) ** 4) / std ** 4
    return kurtosis

def skewness(x):
    """
    Compute the skewness of x.
    :param x: vector x
    :return: skewness of x
    """
    # compute the mean of x
    mean = np.mean(x)
    # compute the standard deviation of x
    std = np.std(x)
    # compute the skewness of x
    skewness = np.mean((x - mean) ** 3) / std ** 3
    return skewness

if __name__ == '__main__':
    X_picked = np.load('X.npy')
    print(X_picked)
    
    # compute kurtosis and skewness of each column of X_picked
    kurtosis_list = [kurtosis(X_picked[:, i]).round(3) for i in range(X_picked.shape[1])]
    skewness_list = [skewness(X_picked[:, i]).round(3) for i in range(X_picked.shape[1])]
    # print the kurtosis and skewness of each column of X_picked
    print('The kurtosis of each column of X_picked is', kurtosis_list)
    print('The skewness of each column of X_picked is', skewness_list)
    # conclude if the data is normal distribution
    print('The data is normal distribution' if np.allclose(kurtosis_list, 3) and np.allclose(skewness_list, 0) else 'The data is not normal distribution')