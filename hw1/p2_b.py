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
    topk_eigenvalues = []
    for _ in range(k):
        eigenvalue, eigenvector = power_iteration(var_cov_matrix)
        topk_PC.append(eigenvector)
        topk_eigenvalues.append(eigenvalue)
        var_cov_matrix = var_cov_matrix - eigenvalue * np.outer(eigenvector, eigenvector) # deflation
    return np.array(topk_PC), np.array(topk_eigenvalues)

if __name__ == '__main__':
    X_picked = np.load('X_picked.npy')
    print(X_picked)
    K = 6
    
    
    # compute the top K principal components of X_picked and print them
    topk_PC, topk_eigv = find_topk_PC(X_picked, k=K)
    print('The top 3 principal components of X_picked are:')
    print(topk_PC.round(2))
    print('The corresponding eigenvalues are:', topk_eigv)
    X_picked_transformed = np.dot(topk_PC, X_picked)
    print('The transformed X_picked is:', X_picked_transformed.round(2))

    # plot cumulative variance explained
    _K = 6
    plt.figure()
    plt.plot(np.cumsum(topk_eigv[:_K]) / np.sum(topk_eigv), '-o')
    print('The cumulative variance explained by the top 3 principal components is:', np.cumsum(topk_eigv[:_K]) / np.sum(topk_eigv))
    # add values of each point on y-axis
    for i, (x, y) in enumerate(zip(np.arange(_K), np.cumsum(topk_eigv[:_K]) / np.sum(topk_eigv))):
        plt.annotate(f'{y:.3f}', (x, y), textcoords='offset pixels', xytext=(-3, -17), ha='left')

    # we only need k bins on x-axis and start from 1 instead of 0
    plt.xticks(ticks=np.arange(_K), labels=np.arange(1, _K+1))
    
    plt.xlabel('number of principal components')
    plt.ylabel('cumulative variance explained')
    plt.title(f'Cumulative variance explained by the top {_K} principal components')
    plt.tight_layout()
    plt.savefig('p2_b_cumulative_variance_explained.png')
    plt.clf()

    # validate the result using sklearn.decomposition.PCA
    pca = PCA(n_components=K)
    pca.fit(X_picked.T)
    print('Computed by sklearn.decomposition.PCA:')
    print(pca.components_.T.round(2))
    
    print(f'topk_PC is correct: {np.allclose(np.abs(topk_PC.T.round(2)), np.abs(pca.components_.T.round(2)))}')


    # plot the transformed X_picked in 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_picked_transformed[0], X_picked_transformed[1], X_picked_transformed[2])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title('Data points projected onto the top 3 principal components')
    plt.tight_layout()
    plt.savefig('p2_b.png')