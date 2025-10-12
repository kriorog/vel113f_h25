# Import pyomo to create the optimization model. 
# Import os if NEOS server is used to access solver.

import pyomo.environ as pyo
import os

# define the pyomo model
model = pyo.ConcreteModel(name="Varme")


# =======================================
# FUNCTIONS
# =======================================
def rule_param_power_demand(model, j):
    # rule function to create the set of power demands for each time period j.

    # input data from problem definition
    demand = {
        1: 50,
        2: 60,
        3: 80,
        4: 70,
        5: 60
    }

    # access correct demand for each time period input j, assuming two con_sequtive schedules are being optimized.
    if j <= 5:
        return demand[j]
    else:
        return demand[j-5]


def rule_con_demand(model, j):
    # rule function for demand constraint, i.e. total power productions needs to at least meet demand in each time period.
    return sum(model.p[k, j] for k in model.power_units) >= model.power_demand[j]


def rule_con_cyclic(model, k, j):
    # rule function for cyclic constraint, require each unit state variable x to be the same in con_sequtive schedules.
    if j <= 5:
        return model.p[k, j] == model.p[k, j+5]
    else:
        return pyo.Constraint.Skip


def rule_con_seq(model, k, j):
    # rule function to constrain each power unit k to not run for more than 3 con_sequtive time periods j.
    # this constraint is only needed for the first half of the time periods, i.e. j = 1,...,5
    # as cyclicity takes care of the other half

    if j <= len(model.time_periods)/2:
        return sum(model.x[k, j] for j in pyo.RangeSet(j, j + 3)) <= 3
    else:
        return pyo.Constraint.Skip


def rule_con_start_lb(model, k, j):
    # rule function for lower bound on start/stop constraint
    if j >= 2:
        return model.x[k, j] - model.x[k, j-1] >= model.y[k, j] - model.z[k, j]
    else:
        return model.x[k, j] >= model.y[k, j]


def rule_con_start_ub(model, k, j):
    # rule function for upper bound on start/stop constraint
    if j >= 2:
        return model.x[k, j] - model.x[k, j-1] <= (1 - model.z[k, j]) - (1 - model.y[k, j])
    else:
        return model.x[k, j] <= model.y[k, j]


def rule_con_warm_start_lb(model, k, j):
    # rule function for lower bound on warm start constraint
    if j >= 2:
        return model.z[k, j-1] + model.y[k, j] >= 2*model.w[k, j]
    else:
        return model.z[k, j] + model.w[k, j] == 0


def rule_con_warm_start_ub(model, k, j):
    # rule function for upper bound on warm start constraint
    if j >= 2:
        return model.z[k, j - 1] + model.y[k, j] <= 2*model.w[k, j] + (1 - model.w[k, j])
    else:
        return model.z[k, j] + model.w[k, j] == 0


def rule_obj_cost(model):
    # rule function for objective function, i.e. minimize total cost of running the power units

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
        for k in model.power_units
    )

    repeat_warm_start_cost = sum(
        model.w[k, j] * model.start_cost[k]
        for k in model.power_units
        for j in list(model.time_periods)[5:]
    )

    repeat_cold_start_cost = sum(
        (
                sum(model.y[k, j] for j in list(model.time_periods)[5:])
                - sum(model.w[k, j] for j in list(model.time_periods)[5:])
        )
        * 1.5
        * model.start_cost[k]
        for k in model.power_units
    )

    running_cost = sum(
        sum(
            model.p[k, j]
            * model.tau[j]
            for j in model.time_periods
        )
        * model.running_cost[k]
        for k in model.power_units
    )

    return (
        running_cost
        + repeat_cold_start_cost
        + repeat_warm_start_cost
        + initial_cold_start_cost
    )


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


# ======================================
# SETS
# ======================================

# define the set of power units
model.power_units = pyo.Set(
    initialize=['M1', 'M2', 'M3']
)

# define the set of time periods, for two consequtive schedules of 5 time periods each
model.time_periods = pyo.RangeSet(1, 10)

# ======================================
# PARAMETERS
# ======================================

# length of each time period
model.tau = pyo.Param(
    model.time_periods,
    domain=pyo.NonNegativeReals,
    initialize=5
)

# start cost of each power unit k
model.start_cost = pyo.Param(
    model.power_units,
    domain=pyo.NonNegativeReals,
    initialize={
        'M1': 10,
        'M2': 13,
        'M3': 16
    }
)

# running cost of each power unit k
model.running_cost = pyo.Param(
    model.power_units,
    domain=pyo.NonNegativeReals,
    initialize={
        'M1': 2.5,
        'M2': 2.5,
        'M3': 2.4
    }
)

# power demand in each time period j
model.power_demand = pyo.Param(
    model.time_periods,
    domain=pyo.NonNegativeReals,
    rule=rule_param_power_demand
)

# lower bound on power output of each power unit k
model.unit_limit_lb = pyo.Param(
    model.power_units,
    initialize={
        'M1': 10,
        'M2': 12,
        'M3': 15
    }
)

# upper bound on power output of each power unit k
model.unit_limit_ub = pyo.Param(
    model.power_units,
    initialize={
        'M1': 50,
        'M2': 40,
        'M3': 55+25 # modified upper limit based on dual analysis
    }
)

# ======================================
# VARIABLES
# ======================================

# x is a binary variable indicating if power unit k is running in time period j
model.x = pyo.Var(
    model.power_units,
    model.time_periods,
    domain=pyo.Binary,
    initialize=0
)

# gamma is a real variable between 0 and 1 describing the power output of unit k in time period j 
# as a fraction of its possible power range

model.gamma = pyo.Var(
    model.power_units,
    model.time_periods,
    domain=pyo.NonNegativeReals,
    bounds=(0, 1)
)

# real variable power output of unit k in time period j
model.p = pyo.Var(
    model.power_units,
    model.time_periods,
    domain=pyo.NonNegativeReals
)

# binary variable y describes if unit k is started in time period j
model.y = pyo.Var(
    model.power_units,
    model.time_periods,
    domain=pyo.Binary,
    initialize=0
)

# binary variable z describes if unit k is stopped in time period j
model.z = pyo.Var(
    model.power_units,
    model.time_periods,
    domain=pyo.Binary,
    initialize=0
)

# binary variable w describes if unit k is started in time period j as a warm start
model.w = pyo.Var(
    model.power_units,
    model.time_periods,
    domain=pyo.Binary,
    initialize=0
)

# ======================================
# CONSTRAINTS
# ======================================

# constraint to define relationship between load ratio gamma to power output p of each unit k in time period j
model.con_gamma = pyo.Constraint(
    model.power_units,
    model.time_periods,
    rule=lambda model, k, j: model.gamma[k, j] <= model.x[k, j]
)

# constraint for upper bound on produced power of each unit k in time period j
model.con_unit_load = pyo.Constraint(
    model.power_units,
    model.time_periods,
    rule=lambda model, k, j: model.p[k, j] == model.x[k, j]*model.unit_limit_lb[k] + model.gamma[k, j]*(model.unit_limit_ub[k] - model.unit_limit_lb[k])
)

# constraint to ensure total power production meets demand in each time period j
model.con_demand = pyo.Constraint(
    model.time_periods,
    rule=rule_con_demand
)

# constraint to ensure cyclicity of the solution
model.con_cyclic = pyo.Constraint(
    model.power_units,
    model.time_periods,
    rule=rule_con_cyclic
)

# constraint to ensure no unit runs for more than 3 consequtive time periods
model.con_seq = pyo.Constraint(
    model.power_units,
    model.time_periods,
    rule=rule_con_seq
)

# start/stop constraint, lower bound
model.con_start_lb = pyo.Constraint(
    model.power_units,
    model.time_periods,
    rule=rule_con_start_lb
)

# start/stop constraint, upper bound
model.con_start_ub = pyo.Constraint(
    model.power_units,
    model.time_periods,
    rule=rule_con_start_ub
)

# constraint to ensure that a unit cannot be started and stopped in the same time period
model.con_start_criteria = pyo.Constraint(
    model.power_units,
    model.time_periods,
    rule=lambda model, k, j: model.y[k, j] + model.z[k, j] <= 1
)

# warm start constraint, upper bound
model.con_warm_start_ub = pyo.Constraint(
    model.power_units,
    model.time_periods,
    rule=rule_con_warm_start_ub
)

# warm start constraint, lower bound
model.con_warm_start_lb = pyo.Constraint(
    model.power_units,
    model.time_periods,
    rule=rule_con_warm_start_lb
)

# ======================================
# OBJECTIVE
# ======================================

# define objective function to minimize total cost
model.obj_cost = pyo.Objective(
    rule=rule_obj_cost,
    sense=pyo.minimize
)


# choose the solver 'cplex' - commercial solver with free academic license - you need to install software from IBM
solver_path = '/opt/homebrew/opt/glpk/bin/glpsol'  # path to glpk solver on mac using homebrew
opt = pyo.SolverFactory('glpk', executable=solver_path)
sol_milp = opt.solve(model, tee=False)


# print output
sol_milp.write()
model.pprint()


# ==================================================
# solve relaxed problem to obtain dual variables
# ==================================================

# fix all integer variables
model.x.fix()
model.y.fix()
model.z.fix()
model.w.fix()

# dual variable suffix to model
model.dual = pyo.Suffix(
    direction=pyo.Suffix.IMPORT_EXPORT
)

sol_lp = opt.solve(model, tee=True) # for local cplex solver
# sol_lp = solver_manager.solve(model, opt = opt)   # using neos
sol_milp.write()
model.pprint()
