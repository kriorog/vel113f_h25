from scipy.optimize import linprog
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html#scipy.optimize.linprog
import numpy as np

# define 4 decision variables, one for the amount of each liquid in litres in the mix.
# x = [x_pils, x_vodka, x_brandy, x_malt]

# define total volume in mix
v_total = 100

# define also the alcohol by volume in each liquid
# p = [p_pils, p_vodka, p_brandy, p_malt]
p = np.array([2.25, 40, 40, 1.5])/100

# define the cost of each liquid per liter
# c = [c_pils, c_vodka, c_brandy, c_malt]
c = np.array([100, 2000, 3000, 120])

# define the upper bound constraints with A_ub and b_ub
A_ub = [[0, 1, 1, 0]]

b_ub = [0.1*v_total]

# define equality constraints with A_eq and b_eq
A_eq = [[1, 1, 1, 1]]
b_eq = [v_total]

# define variable bounds
bounds = [(0,np.inf), (0,7), (2,np.inf), (3,5)]

# use linprog to solve (convert objective to minimization)
OptimizeResult = linprog(-p, A_ub, b_ub, A_eq, b_eq, bounds=bounds, method='simplex')

# print solution
print("Optimal solution: ", OptimizeResult.x)
print("Optimal function value: ", OptimizeResult.fun)
print("Consraint slacks: ", OptimizeResult.slack)
print("Consraint residuals: ", OptimizeResult.con)
print("Success?: ", OptimizeResult.success)
print("Message: ", OptimizeResult.message)