import pyomo.environ as pyo
import os

# define the pyomo model
model = pyo.ConcreteModel(name="Varme")


# =======================================
# FUNCTIONS
# =======================================
def rule_param_power_demand(model, j):

    # define dict with demand in each time period
    demand = {
        1: 50,
        2: 60,
        3: 80,
        4: 70,
        5: 60
    }

    if j <= 5:
        return demand[j]
    else:
        return demand[j-5]


def rule_con_demand(model, j):
    return sum(model.unit_load[k, j] for k in model.units) >= model.power_demand[j]


def rule_con_cyclic(model, k, j):
    if j <= 5:
        return model.unit_load[k, j] == model.unit_load[k, j+5]
    else:
        return pyo.Constraint.Skip


def rule_con_sequence(model, k, j):
    if j <= len(model.time_periods)/2:
        return sum(model.x[k, j] for j in pyo.RangeSet(j, j + 3)) <= 3
    else:
        return pyo.Constraint.Skip


def rule_con_start_lb(model, k, j):
    if j >= 2:
        return model.x[k, j] - model.x[k, j-1] >= model.y[k, j] - model.z[k, j]
    else:
        return model.x[k, j] >= model.y[k, j]


def rule_con_start_ub(model, k, j):
    if j >= 2:
        return model.x[k, j] - model.x[k, j-1] <= (1 - model.z[k, j]) - (1 - model.y[k, j])
    else:
        return model.x[k, j] <= model.y[k, j]


def rule_con_warm_start_lb(model, k, j):
    if j >= 2:
        return model.z[k, j-1] + model.y[k, j] >= 2*model.w[k, j]
    else:
        return model.z[k, j] + model.w[k, j] == 0


def rule_con_warm_start_ub(model, k, j):
    if j >= 2:
        return model.z[k, j - 1] + model.y[k, j] <= 2*model.w[k, j] + (1 - model.w[k, j])
    else:
        return model.z[k, j] + model.w[k, j] == 0


def rule_obj_cost(model):
    initial_cold_start_cost = sum(
        (
           1.5 * model.start_cost[k]
        )
        *
        (
            sum(
                model.y[k, j]
                for j in list(model.time_periods)[:6]
            )
            -
            sum(
                model.y[k, j]
                for j in list(model.time_periods)[5:]
            )
        )
        for k in model.units
    )

    repeat_warm_start_cost = sum(
        model.w[k, j] * model.start_cost[k]
        for k in model.units
        for j in list(model.time_periods)[5:]
    )

    repeat_cold_start_cost = sum(
        (
                sum(model.y[k, j] for j in list(model.time_periods)[5:])
                - sum(model.w[k, j] for j in list(model.time_periods)[5:])
        )
        * 1.5
        * model.start_cost[k]
        for k in model.units
    )

    return repeat_cold_start_cost + repeat_warm_start_cost + initial_cold_start_cost


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


# SETS
model.units = pyo.Set(
    initialize=['M1', 'M2', 'M3']
)

model.time_periods = pyo.RangeSet(1, 10)

# PARAMETERS
model.T = pyo.Param(
    model.time_periods,
    domain=pyo.NonNegativeReals,
    initialize=5
)

model.start_cost = pyo.Param(
    model.units,
    domain=pyo.NonNegativeReals,
    initialize={
        'M1': 10,
        'M2': 13,
        'M3': 16
    }
)

model.running_cost = pyo.Param(
    model.units,
    domain=pyo.NonNegativeReals,
    initialize={
        'M1': 2.5,
        'M2': 2.5,
        'M3': 2.5
    }
)


model.power_demand = pyo.Param(
    model.time_periods,
    domain=pyo.NonNegativeReals,
    rule=rule_param_power_demand
)

model.unit_load_lb = pyo.Param(
    model.units,
    initialize={
        'M1': 10,
        'M2': 12,
        'M3': 15
    }
)

model.unit_load_ub = pyo.Param(
    model.units,
    initialize={
        'M1': 50,
        'M2': 45,
        'M3': 55
    }
)

# VARIABLES
model.x = pyo.Var(
    model.units,
    model.time_periods,
    domain=pyo.Binary,
    initialize=0
)

model.unit_load = pyo.Var(
    model.units,
    model.time_periods,
    domain=pyo.NonNegativeReals
)

model.y = pyo.Var(
    model.units,
    model.time_periods,
    domain=pyo.Binary,
    initialize=0
)

model.z = pyo.Var(
    model.units,
    model.time_periods,
    domain=pyo.Binary,
    initialize=0
)

model.w = pyo.Var(
    model.units,
    model.time_periods,
    domain=pyo.Binary,
    initialize=0
)
# CONSTRAINTS
model.con_unit_load_ub = pyo.Constraint(
    model.units,
    model.time_periods,
    rule=lambda model, k, j: model.unit_load[k, j] <= model.x[k, j]*model.unit_load_ub[k]
)

model.con_unit_load_lb = pyo.Constraint(
    model.units,
    model.time_periods,
    rule=lambda model, k, j: model.unit_load[k, j] >= model.x[k, j]*model.unit_load_lb[k]
)

model.con_demand = pyo.Constraint(
    model.time_periods,
    rule=rule_con_demand
)

model.con_cyclic = pyo.Constraint(
    model.units,
    model.time_periods,
    rule=rule_con_cyclic
)

model.con_sequence = pyo.Constraint(
    model.units,
    model.time_periods,
    rule=rule_con_sequence
)

model.con_start_lb = pyo.Constraint(
    model.units,
    model.time_periods,
    rule=rule_con_start_lb
)

model.con_start_ub = pyo.Constraint(
    model.units,
    model.time_periods,
    rule=rule_con_start_ub
)

model.con_start_criteria = pyo.Constraint(
    model.units,
    model.time_periods,
    rule=lambda model, k, j: model.y[k, j] + model.z[k, j] <= 1
)

model.con_warm_start_ub = pyo.Constraint(
    model.units,
    model.time_periods,
    rule=rule_con_warm_start_ub
)

model.con_warm_start_lb = pyo.Constraint(
    model.units,
    model.time_periods,
    rule=rule_con_warm_start_lb
)

# OBJECTIVE
model.obj_cost = pyo.Objective(
    rule=rule_obj_cost,
    sense=pyo.minimize
)

'''
# choose the solver 'cplex' - commercial solver with free academic license - you need to install software from IBM
solver_path = 'C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio_Community2211\\cplex\\bin\\x64_win64\\cplex.exe'
opt = pyo.SolverFactory('cplex', executable=solver_path)
sol_milp = opt.solve(model, tee=False)
'''

# use neos
solver_manager = pyo.SolverManagerFactory('neos')
os.environ['NEOS_EMAIL'] = 'kristjanor@hi.is'
opt = pyo.SolverFactory('cbc')
sol_milp = solver_manager.solve(model, opt = opt)


# print output
sol_milp.write()
model.pprint()
# print_solution(model)

""" NEXT MODEL """
# now fix the solution and solve relaxed problem
model.x.fix()
model.y.fix()
model.z.fix()
model.w.fix()

# dual variable suffix to model
model.dual = pyo.Suffix(
    direction=pyo.Suffix.EXPORT
)

# sol_lp = opt.solve(model, tee=True)
sol_lp = solver_manager.solve(model, opt = opt)   # using neos
sol_milp.write()
model.pprint()
# print_solution(model)
