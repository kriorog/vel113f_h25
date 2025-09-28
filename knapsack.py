import pyomo.environ as pyo

# problem data and parameters
A = ['hammer', 'wrench', 'screwdriver', 'towel']
b = {'hammer':8, 'wrench':3, 'screwdriver':6, 'towel':11}
w = {'hammer':5, 'wrench':7, 'screwdriver':4, 'towel':3}
W_max = 14

# define the pyomo model
model = pyo.ConcreteModel()

# define variable x for each element in A (tools) representing qty of tool in knapsack
model.x = pyo.Var(A, within=pyo.Binary)

# define the value objective
model.obj = pyo.Objective(
    expr=sum(b[i]*model.x[i] for i in A),
    sense=pyo.maximize
)

# define the weight constraint of the knapsack
model.weight_con = pyo.Constraint(
    expr=sum(w[i]*model.x[i] for i in A) <= W_max
)

# choose the solved 'glpk' - open source solver - you need to install this .exe
solver_path = 'C:\\Program Files (x86)\\glpk-4.65\\w64\\glpsol'
opt = pyo.SolverFactory('glpk', executable=solver_path)

# create a 'dual' suffix component on the instance so the solver plugin will know which suffixes to collect
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

# solve the problem
opt_success = opt.solve(model, tee=True)

# display the output
model.display()

# =============
# Display duals
# =============

# # display all duals
# print("Dual Variables:")
# for constraint in model.component_objects(pyo.Constraint, active=True):
#     dual_value = model.dual[constraint]
#     print(f"{constraint.name}: {dual_value}")