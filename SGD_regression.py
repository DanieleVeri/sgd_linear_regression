import numpy as np


def loss_fn(prediction, y):
    return np.sum((prediction - y)**2)


def stochastic_gradient_descent(x, y, batch_dimension, learning_rate, iterations):
    error_list = []
    x_mat = np.concatenate((np.ones((x.shape[0], 1), dtype=int), x), axis=1)
    coefficients = np.zeros((1+x.shape[1], ), dtype=int)
    for it in range(iterations):
        batch = np.random.randint(len(x), size=batch_dimension)
        x_batch = x_mat[batch, :]
        y_batch = y[batch]
        prediction = np.dot(x_batch, coefficients.T)
        error_list.append(loss_fn(prediction, y_batch))
        gradient = x_batch.T.dot(prediction - y_batch)
        update = -(1/batch_dimension)*learning_rate * gradient
        coefficients = coefficients + update
    return coefficients, error_list
    

def stochastic_gradient_descent_adagrad(x, y, batch_dimension, learning_rate, iterations):
    # initialization of the components of the algorithm
    error_list = []
    g_2 = np.zeros(1+x.shape[1], dtype=int)
    x_mat = np.concatenate((np.ones((x.shape[0], 1), dtype=int), x), axis=1)  # added column of 1 to x for the bias
    coefficients = np.zeros((1+x.shape[1], ), dtype=int)
    for it in range(iterations):
        # random selection of a batch
        batch = np.random.randint(len(x), size=batch_dimension)
        x_batch = x_mat[batch, :]
        y_batch = y[batch]
        # calculations for updating
        prediction = np.dot(x_batch, coefficients.T)
        error_list.append(loss_fn(prediction, y_batch))
        gradient = x_batch.T.dot(prediction - y_batch)
        g_2 = g_2 + gradient*gradient  # adagrad component
        update = -(1/batch_dimension)*learning_rate * g_2**(-1/2) * gradient
        coefficients = coefficients + update
    return coefficients, error_list
