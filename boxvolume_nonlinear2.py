from scipy.optimize import minimize, NonlinearConstraint
import math


def objective(x):
    # box volume V=8x1x2x3
    # use prod function that multiplies all elements in a list/array
    # note that we have to transpose the problem to a minimization problem
    # multiply the volume(x) with -1 and minimize that function
    # can also be done -8*x[0]*x[1]*x[2]
    return -8*math.prod(x)


def constraint_fun(x):
    # define the constraint function over x
    # can also be done x[0]**2+x[1]**2+x[2]**2
    return sum([j**2 for j in x])


# create the constraint, equality constraint =1, same as bounding to be <=1 and >=1
constraint = NonlinearConstraint(constraint_fun, 1, 1)

# initial guess for each variable
x0 = [0.5, 0.5, 0.5]

# create the model and solve
OptimizeResult = minimize(objective, x0, constraints=[constraint], method='trust-constr')
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
# https://pypi.org/project/trust-constr/

# print solution
print("Optimal solution: ", OptimizeResult.x)
print("Optimal function value: ", OptimizeResult.fun)
print("Lagrange multiplier(s): ", OptimizeResult.v)
print("Constraint violation(s): ", OptimizeResult.constr_violation)
print("Objective gradient: ", OptimizeResult.grad)
print("Constraint Jacobian: ", OptimizeResult.jac)
print("Lagrangian gradient: ", OptimizeResult.optimality)
print("Number of evaluations: ", OptimizeResult.nfev)
print("Termination message: ", OptimizeResult.message)
