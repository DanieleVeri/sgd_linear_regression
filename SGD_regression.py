import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_ocean_data():
    df = pd.read_csv('ocean.csv')
    return df[['R_SALINITY', 'R_TEMP']]

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


np.random.seed(10)
iterations = 20000

data = load_ocean_data()
data = data.dropna()
#plt.plot(data['R_TEMP'], data['R_SALINITY'], ".")
#plt.show()

x, y = np.array(data['R_TEMP']), np.array(data['R_SALINITY'])
batch_size = int(len(x) / 1000)

#x, y = points_generator(100, custom_function, 0, 10)
coefficients, error_list = stochastic_gradient_descent(x, y, np.array([0, 0]),
    batch_dimension=batch_size, learning_rate=1e-3, iterations=iterations)
print(coefficients)
print(error_list[iterations-1])

x_mat = np.array([x**i for i in range(len(coefficients))]).T
plt.plot(x, y, '.')
plt.plot(x, np.dot(x_mat, coefficients.T))

plt.figure(2)
plt.plot(np.arange(iterations), error_list)
plt.show()
