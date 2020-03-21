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

def custom_function(x):
    return 8+8*x

def loss_fn(prediction, y):
    return np.sum((prediction - y)**2)

def stochastic_gradient_descent(x, y, coefficients, batch_dimension, learning_rate, iterations):
    error_list = []
    x_mat = np.array([x**i for i in range(len(coefficients))]).T
    for it in range(iterations):
        batch = np.random.randint(len(x), size=batch_dimension)
        X_batch = x_mat[batch, :]
        prediction = np.dot(X_batch, coefficients.T)
        error = loss_fn(prediction, y[batch])
        error_list.append(error)
        gradient = (X_batch.T.dot((prediction - y[batch])))
        adjust = (1/batch_dimension)*learning_rate * gradient
        coefficients = coefficients - adjust
    return coefficients, error_list


iterations = 200000
x, y = points_generator(100, custom_function, 0, 10)
coefficients, error_list = stochastic_gradient_descent(x, y, np.array([0, 0]),
    batch_dimension=20, learning_rate=1e-4, iterations=iterations)
print(coefficients)

x_mat = np.ones(len(x), dtype=int)
for i in range(1, len(coefficients)):
    x_mat = np.vstack((x_mat, x ** i))
x_mat = x_mat.T
plt.plot(x, np.dot(x_mat, coefficients.T))
plt.plot(x, y, '.')
plt.figure(2)
plt.plot(np.arange(iterations), error_list)
plt.show()
