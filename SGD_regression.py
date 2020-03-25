import numpy as np


def loss_fn(prediction, y):
    return np.sum((prediction - y)**2)

def stochastic_gradient_descent_adagrad(x, y, batch_dimension, learning_rate, iterations):
    error_list = []
    diagonal_g = np.zeros(x.shape[1], dtype=int)
    x_mat = np.concatenate((np.ones((x.shape[0], 1), dtype=int), x), axis=1)
    coefficients = np.zeros((1+x.shape[1], ), dtype=int)
    for it in range(iterations):
        batch = np.random.randint(len(x), size=batch_dimension)
        X_batch = x_mat[batch, :]
        prediction = np.dot(X_batch, coefficients.T)
        error = loss_fn(prediction, y[batch])
        error_list.append(error)
        gradient = (X_batch.T.dot((prediction - y[batch])))
        diagonal_g = diagonal_g + np.diag(np.dot(gradient[:, np.newaxis], gradient[:, np.newaxis].T))
        adjust = (1/batch_dimension)*learning_rate * diagonal_g**(-1/2) * gradient
        coefficients = coefficients - adjust
    return coefficients, error_list

