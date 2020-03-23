import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.optimize as op
import ex2_1


def mapfeature(X1, X2):
    degree = 6
    X1.shape = (X1.size, 1)
    X2.shape = (X2.size, 1)
    out = np.ones(shape=(X1[:, 0].size, 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (X1 ** (i - j)) * (X2 ** j)
            out = np.append(out, r, axis=1)
    return out


def costFunc(theta, X, Y ):
    sum = 0
    theta_sum = 0
    for i in range(0, m, 1):
        h = ex2_1.hx(theta, X[i])
        sum += -Y[i] * math.log(h) - (1 - Y[i]) * math.log(1 - h)
    for j in range (1, n+1, 1):
        theta_sum += theta[j]**2
    result = sum/m +lamb/(2*m)
    return result


def cost_grad(theta, X, Y ):
    result = []
    for j in range(len(theta)):
        sum = 0
        for i in range(0, m, 1):
            h = ex2_1.hx(theta, X[i])
            sum += (h - Y[i])*X[i][j]
        if j != 0:
            result.append(sum / m + lamb * theta[j] / m)
        else:
            result.append(sum / m)
    return result


X, Y, admitted_score1, admitted_score2, not_admitted_score1, not_admitted_score2 = ex2_1.visualise_data("ex2data2.txt")

X = mapfeature(X[:, 0], X[:, 1])
m, n = X.shape
lamb = 1
ones = np.ones(m)
X = np.column_stack((ones, X)) # Add a column of ones to x
initial_theta = np.zeros(n+1)

# Compute and display initial cost and gradient
cost = costFunc(initial_theta, X, Y)
grad = cost_grad(initial_theta, X, Y)

print('Cost at initial theta (zeros): {}\n'.format(cost))
print('Gradient at initial theta (zeros): {}\n'.format(grad))

Result = op.minimize(fun=costFunc,
                     x0=initial_theta,
                     args=(X, Y),
                     method='TNC',
                     jac=cost_grad)
optimal_theta = Result.x
print("\nOptimal theta: {}".format(optimal_theta))
print("Cost with optimal theta: {}".format(costFunc(optimal_theta, X, Y)))

#Plot Boundary
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros(shape=(len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        val = mapfeature(np.array(u[i]), np.array(v[j]))
        ones = np.ones(1)
        val = np.column_stack((ones, val))  # Add a column of ones to x
        z[i, j] = val.dot(np.array(optimal_theta))
z = z.T
plt.contour(u, v, z, levels = 0)
plt.title('lambda = {}'.format(lamb))

plt.plot(admitted_score1, admitted_score2, 'bx', label='Admitted')
plt.plot(not_admitted_score1, not_admitted_score2, 'ro', label='Not admitted')
plt.ylabel('Test 2')
plt.xlabel('Test 1')
plt.legend()

plt.show()