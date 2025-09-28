import pyomo.environ as pyo
import numpy as np
import os

# define the pyomo model
model = pyo.ConcreteModel(name="Job shop")

# define a matrix A(j,k) with production time of product j on machine k
A = np.array([
    [3, 0, 10, 6],
    [8, 2, 0, 3],
    [0, 7, 4, 0]
])

# =======================================
# FUNCTIONS
# =======================================


def prod_seq_rule(model, j, l, k):

    if (j in model.K[k].intersection(model.K[l])) and (l != k):
        if model.S[j].ord(l) < model.S[j].ord(k):
            return model.t[j, l] + model.tau[j, l] <= model.t[j, k]
        else:
            return pyo.Constraint.Skip
    else:
        return pyo.Constraint.Skip


def start_seq_rule(model, i, j, k):

    if i != j:
        return model.t[i, k] + model.tau[i, k] <= model.t[j, k] + model.M * (1 - model.x[i, j, k])
    else:
        return pyo.Constraint.Skip


def print_solution(result_model):

    # print model name
    print(f"Model name: {result_model.name}")

    # print objective function value
    for obj in result_model.component_objects(pyo.Objective):
        print(f"Objective name: {obj.name} = {pyo.value(obj)}")

    # print variables and bounds
    for var in result_model.component_objects(pyo.Var):
        for idx in var:
            print(f"Variable name: {var[idx].name}, "
                  f"value = {pyo.value(var[idx])}, "
                  f"lower slack = {var[idx].bounds}"
                  )

    # print constraint function values, slacks, and dual variables
    for con in result_model.component_objects(pyo.Constraint):
        for idx in con:
            # calculate the constraint slack
            # slack = pyo.value(con[idx].upper) - pyo.value(con[idx].body)
            print(f"Constraint name: {con[idx].name}, "
                  f"value = {pyo.value(con[idx])}, "
                  f"lower slack = {con[idx].lslack()}, "
                  f"upper slack = {con[idx].uslack()}, "
                  )

            try:
                print(f"dual variable = {result_model.dual[con[idx]]}")
            except:
                print('Duals are not available. Ensure problem type and/or solver supports dual extraction')

    return result_model

# =======================================


# define sets
model.products = pyo.Set(
    initialize = ['A', 'B', 'C']
)

model.machines = pyo.Set(
    initialize = ['M1', 'M2', 'M3', 'M4']
)

model.K = pyo.Set(
    model.machines,
    initialize = {
        'M1': ['A', 'B'],
        'M2': ['B', 'C'],
        'M3': ['A', 'C'],
        'M4': ['A', 'B']
    }
)

model.S = pyo.Set(
    model.products,
    initialize = {
        'A': ['M1', 'M3', 'M4'],
        'B': ['M1', 'M2', 'M4'],
        'C': ['M2', 'M3']
    }
)

# define parameters
model.tau = pyo.Param(
    model.products,
    model.machines,
    domain = pyo.NonNegativeReals,
    initialize = lambda model, j, k: A[model.products.ord(j)-1, model.machines.ord(k)-1]
)

model.M = pyo.Param(
    domain = pyo.NonNegativeReals,
    initialize = np.sum(A)
)

# define variables
model.x = pyo.Var(
    model.products,
    model.products,
    model.machines,
    within = pyo.Binary,
    initialize = 0
)

model.t = pyo.Var(
    model.products,
    model.machines,
    within = pyo.NonNegativeReals
)

model.T = pyo.Var(
    within = pyo.NonNegativeReals
)


# define constraints
model.con_start_seq = pyo.Constraint(
    model.products,
    model.products,
    model.machines,
    rule = start_seq_rule
)

model.con_symmetry = pyo.Constraint(
    model.products,
    model.products,
    model.machines,
    rule = lambda model, i, j, k: model.x[i, j, k] + model.x[j, i, k] == 1 if i != j else pyo.Constraint.Skip
)

model.total_time = pyo.Constraint(
    model.products,
    model.machines,
    rule = lambda model, j, k: model.t[j, k] + model.tau[j, k] <= model.T if j in model.K[k] else pyo.Constraint.Skip
)

model.prod_seq = pyo.Constraint(
    model.products,
    model.machines,
    model.machines,
    rule = prod_seq_rule    # use rule function because lambda expression is too long
)

# define objective
model.obj_time = pyo.Objective(
    expr = model.T,
    sense = pyo.minimize
)

# choose the solver 'cplex' - commercial solver with free academic license - you need to install software from IBM
solver_path = 'C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio_Community2211\\cplex\\bin\\x64_win64\\cplex.exe'
opt = pyo.SolverFactory('cplex', executable=solver_path)
sol_milp = opt.solve(model, tee = False)

"""# use neos
solver_manager = pyo.SolverManagerFactory('neos')
os.environ['NEOS_EMAIL'] = 'kristjanor@hi.is'
opt = pyo.SolverFactory('glpk')
sol_milp = solver_manager.solve(model, opt = opt)
"""

# print output
sol_milp.write()
model.pprint()
# print_solution(model)

""" NEXT MODEL """
# now fix the solution and solve relaxed problem
model.x.fix()

# dual variable suffix to model
model.dual = pyo.Suffix(
    direction=pyo.Suffix.IMPORT_EXPORT
)

sol_lp = opt.solve(model, tee = False)
# sol_lp = solver_manager.solve(model, opt = opt)   # using neos
sol_milp.write()
model.pprint()
