import pyforest
from p2_a import var_covar_matrix
from p1_b import power_iteration


def find_topk_PC(X, k=3):
    """
    Find the top k principal components of X.
    :param X: matrix X
    :param k: number of principal components
    :return: top k principal components
    """
    # compute the variance-covariance matrix of X
    var_cov_matrix = var_covar_matrix(X)
    
    # compute the top k principal components of X
    topk_PC = []
    for _ in range(k):
        eigenvalue, eigenvector = power_iteration(var_cov_matrix)
        topk_PC.append(eigenvector)
        var_cov_matrix = var_cov_matrix - eigenvalue * np.outer(eigenvector, eigenvector) # deflation
    return np.array(topk_PC)

if __name__ == '__main__':
    X_picked = np.load('X_picked.npy')
    print(X_picked)
    
    # compute the top 3 principal components of X_picked and print them
    top3_PC = find_topk_PC(X_picked)
    print(top3_PC.round(2))
    
    # validate the result using sklearn.decomposition.PCA
    pca = PCA(n_components=3)
    pca.fit(X_picked.T)
    print('Computed by sklearn.decomposition.PCA:')
    print(pca.components_.T.round(2))
    print(f'top3_PC is correct: {np.allclose(top3_PC.T.round(2), pca.components_.T.round(2))}')
    
    # cumulative variance explained
    print('cumulative variance explained:')
    print(np.cumsum(pca.explained_variance_ratio_))