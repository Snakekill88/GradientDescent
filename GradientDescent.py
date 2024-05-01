import numpy as np
import matplotlib.pyplot as plt


def cal_cost(theta, x, y):
    m = len(y)
    predictions = x.dot(theta)
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))
    return cost


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    for it in range(iterations):
        prediction = np.dot(X, theta)
        theta = theta - (1 / m) * learning_rate * (X.T.dot((prediction - y)))
        theta_history[it, :] = theta.T
        cost_history[it] = cal_cost(theta, X, y)

    return theta, cost_history, theta_history


if __name__ == '__main__':
    X = 2 * np.random.rand(100, 1)
    Y = 4 + 3 * X + np.random.randn(100, 1)

    it_lr = [(2000, 0.001), (500, 0.01), (200, 0.05), (100, 0.1)]

    for n in it_lr:
        theta = np.random.randn(2, 1)
        X_b = np.c_[(np.ones((len(X), 1)), X)]
        theta, cost_history, theta_history = gradient_descent(X_b, Y, theta, n[1], n[0])

        print('Theta0:              {:0.3f},\nTheta1:              {:03f}'.format(theta[0][0], theta[1][0]))
        print('Final Cost/MSE:      {:0.3f}', format(cost_history[-1]))

        plt.title('lr={:}'.format(n[1]))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.rcParams['savefig.dpi'] = 500
        plt.plot(X, Y, 'o')
        x1 = np.linspace(0.0, 2.0)
        for i in theta_history:
            y1 = i[0] * x1 + i[1]
            plt.plot(x1, y1, color='red', linewidth='.5', alpha=.3)
        plt.show()
        plt.ylabel('Cost')
        plt.xlabel('# of Iterations')

        plt.title('Iterations={:}'.format(n[0]))
        plt.plot(cost_history, 'o', linewidth=0.01)
        plt.show()
