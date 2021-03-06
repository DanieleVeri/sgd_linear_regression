from SGD_regression import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_energy_data():
    df = pd.read_csv('energy.csv', low_memory=False)
    return df

np.random.seed(10)

print("Loading dataset...")
data = load_energy_data()
data = data.dropna()

x, y = np.array(data.drop(['heating', 'cooling'], axis=1), dtype=float), np.array(data['heating'], dtype=float)

coefficients, error_list = stochastic_gradient_descent(x, y,
    batch_dimension=10,
    learning_rate=1e-6,
    iterations=500)

print("\nSGD linear regression")
print("coefficients:", coefficients)
print("error:", error_list.pop())

plt.figure("error")
plt.plot(np.arange(len(error_list)), error_list)
plt.show()

# With adagrad variant, smaller error with less iterations
coefficients, error_list = stochastic_gradient_descent_adagrad(x, y,
   batch_dimension=10,
   learning_rate=1,
   iterations=100)

print("\nSGD linear regression with adagrad")
print("coefficeints:", coefficients)
print("error:", error_list.pop())

plt.figure("error")
plt.plot(np.arange(len(error_list)), error_list)
plt.show()
