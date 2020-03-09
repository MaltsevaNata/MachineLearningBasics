# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def cost(X, Y, theta):
    m = X.shape[0]
    sum = 0
    for i in range(m):
        th0 = theta[0][0]
        th1 = theta[1][0]
        x0 = X[0][i]
        x1 = X[1][i]
        h = theta[0][0] * X[0][i] + theta[1][0] * X[1][i]
        sum += pow((h - Y[i]), 2)
    result = sum/(2*m)
    return result

def grad_descent(X, Y, theta, alfa):
    m = X.shape[0]
    new_theta = theta
    for j in range(new_theta.shape[0]):
        sum = 0
        for i in range(m):
            h = theta[0][0] * X[0][i] + theta[1][0] * X[1][i]
            sum += (h - Y[i]) * X[j][i]
        new_theta[j][0] = theta[j][0] - alfa * sum / m
    theta = new_theta


X = []
Y = []
with open("ex1data1.txt", "r") as data:
    for line in data:
        str = line.replace('\n', '').split(',')
        X.append(float(str[0]))
        Y.append(float(str[1]))

# subplot 1
plt.plot(X,Y, 'rx')
plt.ylabel('Population of City in 10,000s')
plt.xlabel('Profit in $10,000s')

X = np.array(X)
Y = np.array(Y)
m = X.shape[0]
ones = np.ones(m)
X = np.stack((ones, X)) # Add a column of ones to x

theta = np.zeros([2, 1]) # initialize fitting parameters
iterations = 1500
alpha = 0.001
J = []
theta0 = []
theta1 = []
for it in range(iterations):
    theta0.append(theta[0][0])
    theta1.append(theta[1][0])
    J.append(cost(X, Y, theta))
    grad_descent(X, Y, theta, alpha)

h = []
for i in range(m):
    h.append(theta[0][0] * X[0][i] + theta[1][0] * X[1][i])

plt.plot(X[1], h)

J = np.array(J)
predict1 = theta[0][0] + theta[1][0]* 3.5


# subplot 2
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_trisurf(theta0, theta1, J, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
