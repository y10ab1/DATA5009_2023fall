import pyforest

def pi_estimation(N=10000, h=1, r=1):
    x, y = 0, 0
    pi_hat = 0
    in_square = lambda x, y: (-r <= x <= r) and (-r <= y <= r)
    in_circle = lambda x, y: (x**2 + y**2) <= r**2
    for _ in range(N):
        while True:
            eps_x, eps_y = np.random.uniform(-h, h, 2)
            
            if in_square(x + eps_x, y + eps_y):
                x += eps_x
                y += eps_y
                pi_hat += 1 if in_circle(x, y) else 0
                break
    return 4 * pi_hat / N

def pi_estimation_v2(N=10000, h=1, r=1):
    # Improved version of pi_estimation using metropolis-hastings
    x, y = 0, 0
    pi_hat = 0
    in_square = lambda x, y: (-r <= x <= r) and (-r <= y <= r)
    in_circle = lambda x, y: (x**2 + y**2) <= r**2
    # mh_ratio = lambda x, y, x_new, y_new: in_square(x_new, y_new) and in_circle(x_new, y_new) / in_square(x, y) and in_circle(x, y)

    for _ in range(N):
        eps_x, eps_y = np.random.uniform(-h, h, 2)
        x_new, y_new = x + eps_x, y + eps_y
        if in_square(x_new, y_new):
            x, y = x_new, y_new
        pi_hat += 1 if in_circle(x, y) else 0
            
    return 4 * pi_hat / N

def exp_on_n_and_h():
    N = 10000
    hs = [0.125, 0.25, 0.5, 1, 2, 4]
    ns = [100, 1000, 5000, 10000]
    
    # Plotting the results in same figure
    # visualize the results of different h and n
    fig, ax = plt.subplots(1, 2)
    for h in hs:
        for n in ns:
            ori_pi = pi_estimation(n, h)
            improved_pi = pi_estimation_v2(n, h)
            print(f"Estimate of pi (h={h}, n={n}): {ori_pi}")
            print(f"Estimate of pi (h={h}, n={n}, improved): {improved_pi}")
            ax[0].scatter(np.log2(h), ori_pi, color="blue")
            ax[0].scatter(np.log2(h), improved_pi, color="red")
            ax[1].scatter(np.log10(n), ori_pi, color="blue")
            ax[1].scatter(np.log10(n), improved_pi, color="red")
    ax[0].set_xlabel("h (log scale, base 2)")
    ax[0].set_ylabel("pi")
    ax[0].set_title("pi vs h")
    ax[1].set_xlabel("n (log scale, base 10)")
    ax[1].set_ylabel("pi")
    ax[1].set_title("pi vs n")
    ax[0].legend(["original", "improved"])
    ax[1].legend(["original", "improved"])
    ax[0].hlines(y=np.pi, xmin=-3, xmax=3, color="green")
    ax[1].hlines(y=np.pi, xmin=2, xmax=5, color="green")
    plt.savefig("p3.png")
    

    

def main():
    print(f"Estimate of pi (flawed): {pi_estimation()}")
    print(f"Estimate of pi (improved): {pi_estimation_v2()}")

    exp_on_n_and_h()


if __name__ == "__main__":
    main()