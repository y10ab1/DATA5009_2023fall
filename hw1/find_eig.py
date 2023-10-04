import numpy as np

def power_iteration(A, max_iter=1000, tol=1e-10):
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square")
    
    b_k = np.random.rand(n)
    for _ in range(max_iter):
        # Calculate the matrix-by-vector product
        b_k1 = np.dot(A, b_k)
        
        # Calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)
        
        # Re normalize the vector
        b_k = b_k1 / b_k1_norm
        
        # Check convergence
        if np.linalg.norm(b_k1 - b_k) < tol:
            return b_k1_norm, b_k  # Eigenvalue and eigenvector

    return b_k1_norm, b_k

def deflate(A, eigval, eigvec):
    n = A.shape[0]
    u = eigvec/np.linalg.norm(eigvec)
    P = np.eye(n) - 2 * np.outer(u, u)
    Ad = P @ A @ P
    return Ad

def eig(A):
    eigenvalues = []
    eigenvectors = []
    for _ in range(A.shape[0]):
        eigval, eigvec = power_iteration(A)
        eigenvalues.append(eigval)
        eigenvectors.append(eigvec)
        A = deflate(A, eigval, eigvec)
        
    return eigenvalues, np.column_stack(eigenvectors)


if __name__ == "__main__":
    # Test
    A = np.array([[4, -2], [1, 4]])
    eigenvalues, eigenvectors = eig(A)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)

    # Check using numpy.linalg.eig
    eigenvalues_np, eigenvectors_np = np.linalg.eig(A)
    print(f'The results is correct: {np.allclose(eigenvalues, eigenvalues_np) and np.allclose(eigenvectors, eigenvectors_np)}')
    print("Eigenvalues:", eigenvalues_np)
    print("Eigenvectors:\n", eigenvectors_np)