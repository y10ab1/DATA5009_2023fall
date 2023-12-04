import pulp

# Define the linear programming problem
prob = pulp.LpProblem("Maximize_Profit_for_Fertilizer_Company", pulp.LpMaximize)

# Decision variables
x_A = pulp.LpVariable('x_A', lowBound=5000)  # Minimum production constraint for A
x_B = pulp.LpVariable('x_B', lowBound=0)
x_C = pulp.LpVariable('x_C', lowBound=0)
x_D = pulp.LpVariable('x_D', lowBound=4000)  # Minimum production constraint for D

# Objective function
prob += -5 * x_A + 80 * x_B - 160 * x_C - 15 * x_D

# Constraints
prob += 0.05 * x_A + 0.05 * x_B + 0.10 * x_C + 0.15 * x_D <= 1000, "Nitrates Constraint"
prob += 0.10 * x_A + 0.15 * x_B + 0.20 * x_C + 0.05 * x_D <= 2000, "Phosphates Constraint"
prob += 0.05 * x_A + 0.10 * x_B + 0.10 * x_C + 0.15 * x_D <= 1500, "Potash Constraint"

# Solve the problem
prob.solve()

# Extract and display the solution
solution = {
    'x_A': x_A.varValue,
    'x_B': x_B.varValue,
    'x_C': x_C.varValue,
    'x_D': x_D.varValue,
    'Total Profit': pulp.value(prob.objective)
}

print(solution)
