import pyforest
import scipy
from tqdm import tqdm

SEED = 825
np.random.seed(SEED)


def get_params_a():
    theta = np.random.randint(1, 112)
    a1, a2 = np.random.gamma(10, 10, 2)
    lambda1 = np.random.gamma(3, a1)
    lambda2 = np.random.gamma(3, a2)
    return theta, lambda1, lambda2

def get_params_b():
    theta = np.random.randint(1, 112)
    a1, a2 = np.random.gamma(10, 10, 2)
    lambda1 = np.random.gamma(3, a1)
    alpha = np.exp(np.random.uniform(np.log(1/8), np.log(2)))
    lambda2 = alpha * lambda1
    return theta, lambda1, lambda2

def get_params_c():
    theta = np.random.randint(1, 112)
    a1, a2 = np.random.uniform(0, 100, 2)
    lambda1 = np.random.gamma(3, a1)
    lambda2 = np.random.gamma(3, a2)
    return theta, lambda1, lambda2

def get_synthetic_data(theta_true=60, lambda1_true=2.5, lambda2_true=0.8):
    years = np.arange(1851, 1963)
    disasters = np.zeros(len(years))
    for i, year in enumerate(years):
        if i < theta_true:
            disasters[i] = np.random.poisson(lambda1_true)
        else:
            disasters[i] = np.random.poisson(lambda2_true)
    return disasters

def log_likelihood(data, theta, lambda1, lambda2):
    data_before_theta = data[:int(theta)]
    data_after_theta = data[int(theta):]
    ll_before_theta = np.sum([np.log(scipy.stats.poisson.pmf(k, lambda1)) for k in data_before_theta])
    ll_after_theta = np.sum([np.log(scipy.stats.poisson.pmf(k, lambda2)) for k in data_after_theta])
    return ll_before_theta + ll_after_theta

def metropolis_hastings(data, num_samples, get_params=get_params_a):
    thetas, lambdas1, lambdas2 = [], [], []
    theta, lambda1, lambda2 = get_params()
    current_ll = log_likelihood(data, theta, lambda1, lambda2)

    progress_bar = tqdm(range(num_samples))
    for _ in progress_bar:
        theta_new, lambda1_new, lambda2_new = np.random.normal([theta, lambda1, lambda2], 1)
        proposed_ll = log_likelihood(data, theta_new, lambda1_new, lambda2_new)
        alpha = np.exp(proposed_ll - current_ll)

        if np.random.rand() < alpha:
            theta, lambda1, lambda2 = theta_new, lambda1_new, lambda2_new
            current_ll = proposed_ll

        thetas.append(theta)
        lambdas1.append(lambda1)
        lambdas2.append(lambda2)
        progress_bar.set_description('Running mean of theta: %f, lambda1: %f, lambda2: %f' % (np.mean(thetas), np.mean(lambdas1), np.mean(lambdas2)))

    return thetas, lambdas1, lambdas2

def a():
    data = get_synthetic_data()
    thetas, lambdas1, lambdas2 = metropolis_hastings(data, 5000, get_params=get_params_a)
    #plot histogram in 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].hist(thetas, bins=50, color='orange')
    axes[0].set_title('Theta')
    axes[1].hist(lambdas1, bins=50, color='green')
    axes[1].set_title('Lambda1')
    axes[2].hist(lambdas2, bins=50, color='blue')
    axes[2].set_title('Lambda2')
    plt.savefig('p4_a.png')
    print('Mean of theta: %f, lambda1: %f, lambda2: %f' % (np.mean(thetas), np.mean(lambdas1), np.mean(lambdas2)))
    return thetas, lambdas1, lambdas2

def b():
    data = get_synthetic_data()
    thetas, lambdas1, lambdas2 = metropolis_hastings(data, 5000, get_params=get_params_b)
    #plot histogram in 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].hist(thetas, bins=50, color='orange')
    axes[0].set_title('Theta')
    axes[1].hist(lambdas1, bins=50, color='green')
    axes[1].set_title('Lambda1')
    axes[2].hist(lambdas2, bins=50, color='blue')
    axes[2].set_title('Lambda2')
    plt.savefig('p4_b.png')
    print('Mean of theta: %f, lambda1: %f, lambda2: %f' % (np.mean(thetas), np.mean(lambdas1), np.mean(lambdas2)))    

def c():
    data = get_synthetic_data()
    thetas, lambdas1, lambdas2 = metropolis_hastings(data, 5000, get_params=get_params_c)
    #plot histogram in 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].hist(thetas, bins=50, color='orange')
    axes[0].set_title('Theta')
    axes[1].hist(lambdas1, bins=50, color='green')
    axes[1].set_title('Lambda1')
    axes[2].hist(lambdas2, bins=50, color='blue')
    axes[2].set_title('Lambda2')
    plt.savefig('p4_c.png')
    print('Mean of theta: %f, lambda1: %f, lambda2: %f' % (np.mean(thetas), np.mean(lambdas1), np.mean(lambdas2)))    



def autocorrelation(samples, max_lag):
    def compute_lag(samples, lag):
        n = len(samples)
        mean_sample = np.mean(samples)
        if lag >= n:
            return 0  # Autocorrelation for full length is 0
        numerator = np.sum((samples[:-lag] - mean_sample) * (samples[lag:] - mean_sample))
        denominator = np.sum((samples - mean_sample) ** 2)
        return numerator / denominator

    return [compute_lag(samples, lag) for lag in range(1, max_lag + 1)]


    return [compute_lag(samples, lag) for lag in range(max_lag)]

def effective_sample_size(samples):
    n = len(samples)
    ac_values = autocorrelation(samples, n)
    tau = 1 + 2 * np.sum(ac_values)
    return n / tau


def plot_autocorrelation(samples, max_lag=100, save_path=None):
    ac_values = autocorrelation(samples, max_lag)
    plt.plot(range(max_lag), ac_values)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Plot')
    plt.savefig(save_path)

def e(samples, save_path=None):
    plot_autocorrelation(samples, save_path=save_path)

def f(samples):
    ess = effective_sample_size(samples)
    print(f'Effective Sample Size: {ess}')





if __name__ == "__main__":
    theta_samples, lambda1_samples, lambda2_samples = a()

    plt.clf()
    # Using the results
    e(theta_samples, save_path='p4_e_theta.png')
    f(theta_samples)
    plt.clf()

    e(lambda1_samples, save_path='p4_e_lambda1.png')
    f(lambda1_samples)
    plt.clf()

    e(lambda2_samples, save_path='p4_e_lambda2.png')
    f(lambda2_samples)
    
    # b()
    # c()