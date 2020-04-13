import numpy as np
import scipy.io
import scipy.optimize as op
from scipy import misc
import matplotlib.pyplot as plt
import sys
import math
sys.path.append("../machine-learning-ex2")
from ex2_1 import sigmoid


def hx(theta, X):
    #theta_t = theta.transpose()
    mult = []
    for i in range(n):
        mult.append(theta@X[i])
    mult = np.array(mult)
    h = sigmoid(mult)
    return h

def cost_grad(theta, X, Y ): # скорее всего неправильно, разобраться с размерностями
    h = hx(theta, X)
    sub = h - Y
    derivative = np.zeros(n)
    for i in range(n):
        Xi = X[i]
        derivative[i] = sum(Xi*sub[i])
        if i>= 1:
            derivative[i]+=(sum(theta**2))*lamb/(2*m)
    result = derivative/m
    return result

def costFunc(theta, X, Y ):
    h = hx(theta, X)
    #sub = h - (1-Y)
    cost = sum(-Y*np.log(h)-(1-Y)*np.log(1-h)) + (sum(theta[1:m]**2))*lamb/(2*m)
    result = cost/m
    return result

def display_image(mat, number):
    image = np.asarray(mat['X'][number], dtype=np.float).reshape(20, 20)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

def predict(theta, X):
    # Add a column of ones to x
    h = hx(theta, X)
    return h

mat = scipy.io.loadmat('ex3data1.mat')
X = np.asarray(mat['X'], dtype=np.float).reshape(5000, 400)
Y = np.asarray(mat['y'], dtype=np.float).reshape(5000, )
n, m = X.shape
ones = np.ones(m)
#X = np.column_stack((ones, X))  # Add a column of ones to x
lamb = 0.01
K = 10
initial_theta = np.zeros(m)
grad = cost_grad(initial_theta, X, Y)
print(grad)
cost = costFunc(initial_theta, X, Y)
print(cost)
#display_image(mat, 0)

theta = np.zeros((n, m))
class_size = n/K
for k in range(K):
    klass = k if k>0 else 10
    y = (Y==klass)
    y = y.astype(float)
    optimal = op.fmin_tnc(func=costFunc,
                x0=initial_theta,
                          fprime = grad,
                          approx_grad=True,
                args=(X, y)
                )
    print(optimal)
    theta[k] = optimal

print(theta)

pr = predict(theta, X[100])
print("\nPredicted value for {} is: {}".format(0, pr))