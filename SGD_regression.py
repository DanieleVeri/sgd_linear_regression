import numpy as np
import matplotlib.pyplot as plt


def points_generator(quantity, function, amplitude_noise):
    np.random.seed(10)
    X = np.arange(quantity)
    y = function(X) + (np.random.rand(quantity,)-0.5)*2*amplitude_noise
    return X, y


def custom_function(X):
    return 10+8*X+5*X**2


def stochastic_gradient_descent(X, y, theta, batch_dimension=10, learning_rate=0.00001, iterations=1000):
    error_list = []
    x_mat = np.array([X**i for i in range(len(theta))]).T
    for it in range(iterations):
        batch = np.random.randint(len(X), size=batch_dimension)
        X_batch = x_mat[batch,:]
        prediction = np.dot(X_batch,theta.T)
        error_list.append(np.sum(prediction - y[batch]))
        adjust = (1/batch_dimension)*learning_rate*(X_batch.T.dot((prediction - y[batch])))
        #print(adjust)
        theta = theta - adjust
    return theta, error_list



iter = 100
X, y = points_generator(50, custom_function, 5e2)
theta, error_list = stochastic_gradient_descent(X, y, np.array([2,1,3]), batch_dimension=20, learning_rate=1e-6, iterations=iter)
print(theta)


x_mat = np.array([X**i for i in range(len(theta))]).T
plt.plot(X,np.dot(x_mat,theta.T))
plt.plot(X,y,'.')
plt.figure(2)
plt.plot(np.arange(iter), error_list)
plt.show()
