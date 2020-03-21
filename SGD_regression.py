import numpy as np
import matplotlib.pyplot as plt


def points_generator(quantity, function, amplitude_noise):
    np.random.seed(10)
    X = np.arange(quantity)
    noise = (np.random.rand(quantity)-0.5) * 2*amplitude_noise
    y = function(X) + noise
    return X, y


def custom_function(x):
    return 100 + 8*x


def stochastic_gradient_descent(x, y, coefficients, batch_dimension, learning_rate, iterations):
    error_list = []
    for it in range(iterations):
        batch = np.random.randint(len(x), size=batch_dimension)
        x_batch = x[batch]
        #print(x_batch)
        prediction = coefficients[0] + coefficients[1] * x_batch
        print(y[batch])
        print(prediction)
        input()
        error_list.append(np.sum(prediction - y[batch]))

        adjust = (1/batch_dimension) * learning_rate * (x_batch.T.dot((prediction - y[batch])))
        #print(adjust)
        coefficients = coefficients - adjust
    return coefficients, error_list


iterations = 10000
x, y = points_generator(50, custom_function, 10)
coefficients, error_list = stochastic_gradient_descent(x, y, np.array([0, 0]),
    batch_dimension=20, learning_rate=1e-6, iterations=iterations)
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
