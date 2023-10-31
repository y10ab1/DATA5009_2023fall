import pyforest

def get_params():
    theta = np.random.uniform(1, 111) # 1 <= theta <= 111
    a1, a2 = np.random.gamma(10, 10, 2) # a1, a2 ~ Gamma(10, 10)
    lambda1 = np.random.gamma(3, a1) # lambda1 ~ Gamma(3, a1)
    lambda2 = np.random.gamma(3, a2) # lambda2 ~ Gamma(3, a2)
    return theta, lambda1, lambda2
    
def main():
    theta, lambda1, lambda2 = get_params()
    print("theta: ", theta, "lambda1: ", lambda1, "lambda2: ", lambda2)

if __name__ == "__main__":
    main()