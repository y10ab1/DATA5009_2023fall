import pyforest



def f(x):
    return np.exp(-np.abs(x)**3 / 3)

def g(x, lambda_val=1):  
    return lambda_val * np.exp(-lambda_val * x)

def importance_sampling_estimate(N=10000, lambda_val=1):
    # 1. Sample from g(x)
    samples_g = np.random.exponential(1/lambda_val, N)

    # 2. Compute weights
    weights = f(samples_g) / g(samples_g, lambda_val)

    # 3. Standardize weights
    weights /= weights.sum()

    # 4. Estimate E(X^2)
    E_X2 = np.sum(weights * samples_g**2)

    return E_X2

def rejection_sampling(f, g, c, size=1, lambda_val=1):
    samples = []
    while len(samples) < size:
        x = np.random.exponential(1/lambda_val)  # Sample from g(x)
        u = np.random.uniform(0, 1)
        
        if u <= f(x) / (c * g(x, lambda_val)):
            samples.append(x)
    
    return np.array(samples)

def rejection_sampling_estimate(N=10000, c=2, lambda_val=1):
    samples = rejection_sampling(f, g, c, N, lambda_val)
    # Estimate E(X^2) using the accepted samples
    E_X2_rejection = np.mean(samples**2)

    return E_X2_rejection


def main():
    print(f"Importance sampling estimate: {importance_sampling_estimate()}")
    print(f"Rejection sampling estimate: {rejection_sampling_estimate()}")

if __name__ == "__main__":
    main()