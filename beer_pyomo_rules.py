import pyomo.environ as pyo
import numpy as np

# define the pyomo model
model = pyo.ConcreteModel(name="Max alcohol mix")

# problem data and parameters
model.A = pyo.Set(initialize=['pilsner', 'vodka', 'brandy', 'malt'], ordered=True)
model.p = pyo.Param(model.A, initialize={'pilsner': 2.25/100, 'vodka': 40/100, 'brandy': 40/100, 'malt': 1.5/100})
model.c = pyo.Param(model.A, initialize={'pilsner': 100, 'vodka': 2000, 'brandy': 3000, 'malt': 120})

# define a subset of strong liquour
# model.strongAlc = pyo.Set(within=model.A, initialize={'vodka', 'brandy'}) # also possible
model.strongAlc = pyo.Set(within=model.A, initialize=list(model.A)[1:3])

# strong alcohol limit
model.maxStrongAlc = pyo.Param(initialize=10)

# define total volume in mix
model.totalVol = pyo.Param(initialize=100)

# define variable bounds
ub = {'pilsner': np.inf, 'vodka': 7, 'brandy': np.inf, 'malt': 5}
lb = {'pilsner': 0, 'vodka': 0, 'brandy': 2, 'malt': 3}


# =======================================
# FUNCTIONS
# =======================================
def bounds_rule(model, j):
    return lb[j], ub[j]


def obj_maxAlc_rule(model):
    return sum(model.p[j]*model.x[j] for j in model.A)


def con_maxStrongAlc_rule(model):
    expr = 0
    for j in model.A:
        if j in model.strongAlc:
            expr += model.x[j]

    return expr <= model.maxStrongAlc

def con_totalVol_rule(model):
    return sum(model.x[j] for j in model.A) == model.totalVol


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
                  f"dual variable = {result_model.dual[con[idx]]}"
                  )

    return result_model

# =======================================


# define variable x for each element in A - describing the amount of each material in litres
model.x = pyo.Var(model.A, within=pyo.Reals, bounds=bounds_rule)
# model.x = pyo.Var(model.A, domain=pyo.Reals, bounds=lambda model, j: (lb[j], ub[j])) # also possible

# define the value objective value for maximum alcohol strength
model.obj_maxAlc = pyo.Objective(rule=obj_maxAlc_rule, sense=pyo.maximize)

# define the constraint on strong alcohol
model.con_maxStrongAlc = pyo.Constraint(rule=con_maxStrongAlc_rule)

# define total volume constraint
model.con_totalVol = pyo.Constraint(rule=con_totalVol_rule)

# choose the solved 'glpk' - open source solver - you need to install this .exe
solver_path = 'C:\\Program Files (x86)\\glpk-4.65\\w64\\glpsol'
opt = pyo.SolverFactory('glpk', executable=solver_path)

# create a 'dual' suffix component on the instance so the solver plugin will know which suffixes to collect
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

# solve the model
sol_maxAlc = opt.solve(model, tee=False)
# sol_maxAlc.write()
# model.pprint()
print_solution(model)

# =====================================
# Now change model to find a 4% beer
# =====================================

# change the model name
model.name = "Min cost mix"

# deactivate the previous objective
model.obj_maxAlc.deactivate()

# define the new objective of minimum cost
model.obj_minCost = pyo.Objective(
    expr=pyo.sum_product(model.c, model.x),
    sense=pyo.minimize
)

# define a new constraint of 4% mix
model.con_mix = pyo.Constraint(
    expr=pyo.sum_product(model.p, model.x) == 4
)

# solve the modified model
sol_minCost = opt.solve(model, tee=False)
# sol_minCost.write()
# model.pprint()
print_solution(model)
