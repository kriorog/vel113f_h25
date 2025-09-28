import numpy as np
from scipy.optimize import fsolve # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html
from autograd import grad # https://github.com/HIPS/autograd
import math


def volume(x):
    # box volume V=8x1x2x3
    # use prod function that multiplies all elements in a list/array
    # note that we have to transpose the problem to a minimization problem
    return 8*math.prod(x)


def constraint_fun(x):
    # define the constraint function over x
    # can also be done x[0]**2+x[1]**2+x[2]**2
    return sum([j**2 for j in x])-1


def Lagrangian(x):
    # define the Lagrange function based on the volume and constraint
    return volume(x[0:3]) - x[3]*constraint_fun(x[0:3])


def gradL(x):
    # gradient of the Lagrangian
    dL=grad(Lagrangian)
    dLdx1, dLdx2, dLdx3, dLdlambd = dL(x)
    return [dLdx1, dLdx2, dLdx3, constraint_fun(x[0:3])]


# initial guess for each variable
x0 = np.array([0.5, 0.5, 0.5, 0.5])

# create the model and solve
Fsol = fsolve(gradL, x0, full_output=1)

# extract the solution
sol = Fsol[0]
print('solution variables', sol)

# compute the residual
res = Fsol[1]['fvec']
print('residual', res)

# what is the optimal volume?
print('Optimal volume', volume(sol[0:3]))
