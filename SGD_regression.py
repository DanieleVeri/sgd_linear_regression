import numpy as np
import matplotlib.pyplot as plt


def points_generator(quantity, function, precision, amplitude_noise, only_positive=True):
    np.random.seed(10)
    limit = quantity/(2*(10**-precision))
    if only_positive:
        X = np.arange(0, 2*limit, 1 * 10 ** precision)
    else:
        X = np.arange(-limit, limit, 1 * 10 ** precision)
    y = function(X) + (np.random.rand(quantity)-0.5)*2*amplitude_noise
    return X, y


def custom_function(X):
    return 100-8*X


def stochastic_gradient_descent(X, y, theta, batch_dimension=10, learning_rate=0.00001, iterations=1000, momentum=0):
    error_list = []
    adjust_list = [theta*0]
    x_mat = np.array([X**i for i in range(len(theta))]).T
    for it in range(iterations):
        batch = np.random.randint(len(X), size=batch_dimension)
        X_batch = x_mat[batch, :]
        prediction = np.dot(X_batch, theta.T)
        error_list.append(np.sum(prediction - y[batch]))
        adjust = (1/batch_dimension)*learning_rate*(X_batch.T.dot((prediction - y[batch]))) + momentum*adjust_list[it-1]
        theta = theta - adjust
        adjust_list.append(adjust)
    return theta, error_list, adjust_list


iter = 10000
X, y = points_generator(int(1e2), custom_function, -0.5, 0)
theta, error_list, adjust_list = stochastic_gradient_descent(X, y, np.array([0, 0]), batch_dimension=20, learning_rate=1e-10, iterations=iter, momentum=0.2)
print(theta)
adjust_list = np.array(adjust_list)

x_mat = np.array([X**i for i in range(len(theta))]).T
plt.plot(X, np.dot(x_mat, theta.T))
plt.plot(X, y, '.')
plt.figure(2)
plt.plot(np.arange(iter), error_list)
plt.figure(3)
plt.plot(np.arange(iter+1), adjust_list)
plt.show()