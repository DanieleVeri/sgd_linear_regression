from SGD_regression import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_ocean_data():
    df = pd.read_csv('ocean.csv', low_memory=False)
    return df[['R_SALINITY', 'R_TEMP']]

np.random.seed(10)

data = load_ocean_data()
data = data.dropna()

x, y = np.array(data[['R_TEMP']], dtype=float), np.array(data['R_SALINITY'], dtype=float)

coefficients, error_list = stochastic_gradient_descent(x, y,
    batch_dimension=int(len(x) / 10000),
    learning_rate=1e-2,
    iterations=5000)

# With adagrad variant, same error with half iterations
#coefficients, error_list = stochastic_gradient_descent_adagrad(x, y,
#    batch_dimension=int(len(x) / 10000),
#    learning_rate=50,
#    iterations=1000)


print(coefficients)
print(error_list.pop())

x_mat = np.concatenate((np.ones((x.shape[0], 1), dtype=int), x), axis=1)
plt.plot(x, y.T, '.')
plt.plot(x, np.dot(x_mat, coefficients.T))

plt.figure(2)
plt.plot(np.arange(len(error_list)), error_list)
plt.show()
