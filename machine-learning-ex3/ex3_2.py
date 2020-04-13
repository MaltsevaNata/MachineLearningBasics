import numpy as np
import scipy.io
import scipy.optimize as op
from scipy import misc
import matplotlib.pyplot as plt
import sys
import math
sys.path.append("../machine-learning-ex2")
from ex2_1 import sigmoid

def predict(Theta1, Theta2, X):
    X = X.transpose()
    z2 = Theta1 @ X
    a2 = sigmoid(z2)
    ones = np.ones(m)
    a2 = a2.transpose()
    a2 = np.column_stack((ones,a2))
    z3 = Theta2 @ a2.transpose()
    a3 = sigmoid(z3)
    predict = np.argmax(a3, axis=0) + 1
    return predict


mat = scipy.io.loadmat('ex3data1.mat')
X = np.asarray(mat['X'], dtype=np.float).reshape(5000, 400)
Y = np.asarray(mat['y'], dtype=np.float).reshape(5000, )
param = scipy.io.loadmat('ex3weights.mat')
Theta1 = np.asarray(param['Theta1'], dtype=np.float).reshape(25, 401)
Theta2 = np.asarray(param['Theta2'], dtype=np.float).reshape(10, 26)
m,n = X.shape # m = 5000
ones = np.ones(m)

X = np.column_stack((ones, X))
res = predict(Theta1, Theta2, X)
compare = (res == Y)
accuracy = sum(compare)*100/m
print("Accuracy = {}" .format(accuracy))

