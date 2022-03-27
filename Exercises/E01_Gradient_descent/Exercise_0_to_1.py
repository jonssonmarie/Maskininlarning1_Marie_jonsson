import numpy as np
import matplotlib.pyplot as plt


# Exercise 0 Simulate dataset
def simulate_samples(samples):
    """
    simulate samples from Normal distribution and Uniform distribution
    :param samples: int
    :return: np.array X = shape (1000, 3), np.array y shape = (1000,)
    """
    np.random.seed(42)
    x = np.random.randn(samples, 2)
    epsilon = np.random.normal(loc=0, scale=1, size=samples)
    y = 3 * x[:, 0] + 5 * x[:, 1] + 3 + epsilon
    one = np.ones(samples)
    X = np.c_[one, x]
    return X, y


X, y = simulate_samples(1000)
y = np.array([y]).T


# Excercise 1 Gradient descent - learning rate
def gradient_descent(X, y, iterations, learning_rate=.1):
    """
    Calculates batch gradient descent.
    :param X: np.array
    :param y: np.array
    :param iterations: int
    :param learning_rate: float
    :return: np.array Theta
    """

    m = len(X)
    # Random number from normal distribution dim (3, 1). Theta is initialised to this value.
    theta = np.random.randn(X.shape[1], 1)

    for _ in range(iterations):
        gradient = 2 / m * X.T @ (X @ theta - y)    # Formula for calculating batch gradient descent
        theta -= learning_rate * gradient           # Decrease and update theta with the learning rate * gradient

    return theta


def iterations_of_theta_values(X, y, learning_rate, max_iterations, step):
    """
    Calculates theta
    :param X: np.array
    :param y: np.array
    :param learning_rate: float
    :param max_iterations: int
    :param step: int
    :return: list of theta values
    """

    theta_values_iterations = []

    for iteration in range(1, max_iterations, step):
        theta_values = gradient_descent(X, y, iteration, learning_rate)
        theta_values = [theta_value for theta_value in theta_values.reshape(-1)]
        theta_values_iterations.append(theta_values)

    return theta_values_iterations


theta_values_500 = iterations_of_theta_values(X, y, 0.1, 501, 1)
theta_values_5000 = iterations_of_theta_values(X, y, 0.01, 5000, 20)
theta_values_5000_1 = iterations_of_theta_values(X, y, 0.001, 5000, 20)


def plot_theta(theta_values, max_iteration, step, title):
    fig, ax = plt.figure(dpi=100), plt.axes()
    _ = ax.plot(range(0, max_iteration, step), theta_values)
    _ = ax.set(title=title, xlabel="Iterations", ylabel="θ-values")
    _ = ax.legend(labels=[f"$θ_{0}$", f"$θ_{1}$", f"$θ_{2}$"])
    plt.show()


plot_theta(theta_values_500, 500, 1, "Gradient Descent, Learning Rate 0.1")
plot_theta(theta_values_5000, 5000, 20, "Gradient Descent, Learning Rate 0.01")
plot_theta(theta_values_5000_1, 5000, 20, "Gradient Descent, Learning Rate 0.001")