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

def main():
    print(f"Estimate of pi (flawed): {pi_estimation()}")
    print(f"Estimate of pi (improved): {pi_estimation_v2()}")

if __name__ == "__main__":
    main()