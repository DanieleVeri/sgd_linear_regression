import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_ocean_data():
    df = pd.read_csv('ocean.csv', low_memory=False)
    return df[['R_SALINITY', 'R_TEMP']]


def loss_fn(prediction, y):
    return np.sum((prediction - y)**2)


def stochastic_gradient_descent(x, y, batch_dimension, learning_rate, iterations):
    error_list = []
    x_mat = np.concatenate((np.ones((x.shape[0], 1), dtype=int), x), axis=1)
    coefficients = np.zeros((1+x.shape[1], ), dtype=int)
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


np.random.seed(10)
iterations = 20000

data = load_ocean_data()
data = data.dropna()
x, y = np.array(data[['R_TEMP']], dtype=float), np.array(data['R_SALINITY'], dtype=float)
batch_size = int(len(x) / 1000)

coefficients, error_list = stochastic_gradient_descent(x, y, batch_dimension=batch_size,
                                                       learning_rate=1e-3, iterations=iterations)
print(coefficients)
print(error_list[iterations-1])

x_mat = np.concatenate((np.ones((x.shape[0], 1), dtype=int), x), axis=1)
plt.plot(x, y.T, '.')
plt.plot(x, np.dot(x_mat, coefficients.T))

plt.figure(2)
plt.plot(np.arange(iterations), error_list)
plt.show()
