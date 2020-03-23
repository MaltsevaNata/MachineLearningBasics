import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.optimize as op


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hx(theta, x):
    theta_t = theta.transpose()
    mult = theta_t.dot(x)
    h = sigmoid(mult)
    return h


def costFunc(theta, X, Y ):
    sum = 0
    for i in range(0, m, 1):
        h = hx(theta, X[i])
        sum += -Y[i] * math.log(h) - (1 - Y[i]) * math.log(1 - h)
    result = sum/m
    return result


def cost_grad(theta, X, Y ):
    result = []
    for j in range(len(theta)):
        sum = 0
        for i in range(0, m, 1):
            h = hx(theta, X[i])
            sum += (h - Y[i])*X[i][j]
        result.append(sum / m)
    return result

def predict(theta, X):
    X = np.concatenate(([1], X),axis=0)  # Add a column of ones to x
    h = hx(theta, X)
    res = int(h + (0.5))
    return res


def visualise_data(file):
    X = []
    Y = []
    admitted_score1 = []
    admitted_score2 = []
    not_admitted_score1 = []
    not_admitted_score2 = []
    with open(file, "r") as data:
        for line in data:
            str = line.replace('\n', '').split(',')
            X.append([float(str[0]), float(str[1])])
            Y.append(int(str[2]))
            if int(str[2]) == 1:
                admitted_score1.append(float(str[0]))
                admitted_score2.append(float(str[1]))
            else:
                not_admitted_score1.append(float(str[0]))
                not_admitted_score2.append(float(str[1]))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y, admitted_score1, admitted_score2, not_admitted_score1, not_admitted_score2


if __name__ == "__main__":
    X, Y, admitted_score1, admitted_score2, not_admitted_score1, not_admitted_score2 = visualise_data("ex2data1.txt")
    m, n = X.shape
    ones = np.ones(m)
    X = np.column_stack((ones, X)) # Add a column of ones to x
    initial_theta = np.zeros(n + 1)

    # Compute and display initial cost and gradient
    cost = costFunc(initial_theta, X, Y)
    grad = cost_grad(initial_theta, X, Y)

    print('Cost at initial theta (zeros): {}\n'.format(cost))
    print('Expected cost (approx): 0.693\n')
    print('Gradient at initial theta (zeros): {}\n'.format(grad))

    print('Expected gradients (approx): -0.1000 -12.0092 -11.2628\n')

    Result = op.minimize(fun = costFunc,
                                     x0 = initial_theta,
                                     args = (X, Y),
                                     method = 'TNC',
                                     jac = cost_grad)
    optimal_theta = Result.x
    print("\nOptimal theta: {}".format(optimal_theta))
    print("Expected cost with optimal theta: {}".format(0.203))
    print("Cost with optimal theta: {}".format(costFunc(optimal_theta, X, Y)))

    scores_to_predict = np.array([45, 85])
    pr = predict(optimal_theta, scores_to_predict)
    print("\nPredicted value for {} is: {}".format(scores_to_predict, pr))

    plot_x = np.array([min(X[:,2])-2, max(X[:,2])+2])
    # Calculate the decision boundary line
    plot_y = (-1/optimal_theta[2])*(optimal_theta[1]*plot_x + optimal_theta[0])

    plt.plot(admitted_score1, admitted_score2, 'bx', label = 'Admitted')
    plt.plot(not_admitted_score1, not_admitted_score2, 'ro', label = 'Not admitted')
    plt.plot(plot_x, plot_y, label = 'Decision boundary')
    plt.ylabel('score2')
    plt.xlabel('score1')
    plt.legend()

    plt.show()