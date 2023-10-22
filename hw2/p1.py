import pyforest


# (a) Normal(μ, σ^2)
def generate_normal(mu, sigma):
    z1 = np.sqrt(-2 * np.log(np.random.uniform(0, 1))) * np.cos(2 * np.pi * np.random.uniform(0, 1))
    z2 = np.sqrt(-2 * np.log(np.random.uniform(0, 1))) * np.sin(2 * np.pi * np.random.uniform(0, 1))
    return mu + sigma * z1  # or z2

# (b) Exponential(λ)
def generate_exponential(lam):
    return -np.log(1 - np.random.uniform(0, 1)) / lam

# (c) Poisson(λ)
def generate_poisson(lam):
    L = np.exp(-lam)
    k = 0
    p = 1
    while True:
        k = k + 1
        p = p * np.random.uniform(0, 1)
        if p <= L:
            break
    return k - 1

# (d) Chi-Square(df=k)
def generate_chi_square(k):
    return sum([(np.sqrt(-2 * np.log(np.random.uniform(0, 1))) * np.cos(2 * np.pi * np.random.uniform(0, 1)))**2 for _ in range(k)])

# (e) F distribution for F_{k, m}
def generate_f(k, m):
    return (generate_chi_square(k) / k) / (generate_chi_square(m) / m)

# (f) Binomial(n, p)
def generate_binomial(n, p):
    return sum([1 if np.random.uniform(0, 1) < p else 0 for _ in range(n)])

# (g) Negative Binomial(r, p)
def generate_negative_binomial(r, p):
    return sum([generate_geometric(p) for _ in range(r)])

# (h) Geometric(p)
def generate_geometric(p):
    return int(np.ceil(np.log(np.random.uniform(0, 1)) / np.log(1 - p)))

# (i) Dirichlet(α1, ... , αk)
def generate_dirichlet(*alpha):
    samples = [np.random.gamma(a, 1) for a in alpha]
    return [s/sum(samples) for s in samples]


    
def main():
    print(f"Normal random variable: {generate_normal(0, 1)}")
    print(f"Exponential random variable: {generate_exponential(1)}")
    print(f"Poisson random variable: {generate_poisson(1)}")
    print(f"Chi-Square random variable: {generate_chi_square(10)}")
    print(f"F random variable: {generate_f(10, 10)}")
    print(f"Binomial random variable: {generate_binomial(10, 0.5)}")
    print(f"Negative Binomial random variable: {generate_negative_binomial(10, 0.5)}")
    print(f"Dirichlet random variable: {generate_dirichlet(1, 2, 3)}")
    pass
    
if __name__ == "__main__":
    main()