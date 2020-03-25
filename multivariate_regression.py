from SGD_regression import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_energy_data():
    df = pd.read_csv('energy.csv', low_memory=False)
    return df

np.random.seed(10)

data = load_energy_data()
data = data.dropna()

x, y = np.array(data.drop(['heating', 'cooling'], axis=1), dtype=float), np.array(data['heating'], dtype=float)

coefficients, error_list = stochastic_gradient_descent_adagrad(x, y,
    batch_dimension=10,                                  
    learning_rate=50,
    iterations=500)

print(coefficients)
print(error_list.pop())

x_mat = np.concatenate((np.ones((x.shape[0], 1), dtype=int), x), axis=1)
plt.plot(x, y.T, '.')
plt.plot(x, np.dot(x_mat, coefficients.T))

plt.figure(2)
plt.plot(np.arange(len(error_list)), error_list)
plt.show()
