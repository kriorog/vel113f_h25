import numpy as np
from scipy.optimize import fsolve  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html


def f(x):
    # x = [x1, x2, x3, lambda]
    return 8 * x[0] * x[1] * x[2]


def Ldiff(x):
    # x = [x1, x2, x3, lambda]
    dL = np.zeros(4)

    # calculate the outputs
    dL[0] = 8 * x[1] * x[2] - 2 * x[3] * x[0]
    dL[1] = 8 * x[0] * x[2] - 2 * x[3] * x[1]
    dL[2] = 8 * x[0] * x[1] - 2 * x[3] * x[2]
    dL[3] = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1

    return dL


# We need to define an initial guess at a solution x0
x0 = np.array([0.5, 0.5, 0.5, 0.5])

# solve the problem
Fsol = fsolve(Ldiff, x0, full_output=1)

# extract the solution
sol = Fsol[0]
print('solution variables', sol)

# compute the residual
res = Fsol[1]['fvec']
print('residual', res)

# what is the optimal volume?
print('Optimal volume', f(sol))