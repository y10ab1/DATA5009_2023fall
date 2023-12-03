import pulp

# Define the problem
prob = pulp.LpProblem("Maximize_Profit", pulp.LpMaximize)

# Variables
x_A = pulp.LpVariable('x_A', lowBound=0)
x_B = pulp.LpVariable('x_B', lowBound=0)
x_C = pulp.LpVariable('x_C', lowBound=0)
x_D = pulp.LpVariable('x_D', lowBound=0)

# Objective Function (example profits)
prob += 20*x_A + 15*x_B + 25*x_C + 30*x_D

# Constraints (example constraints)
prob += 3*x_A + 2*x_B + 4*x_C + 5*x_D <= 1000  # Nitrate
prob += 2*x_A + 3*x_B + 0*x_C + 2*x_D <= 800   # Phosphate
prob += 1*x_A + 0*x_B + 2*x_C + 3*x_D <= 500   # Potash
# Add minimum production constraints for A and D if given

# Solve the problem
prob.solve()

# Print the results
for v in prob.variables():
    print(v.name, "=", v.varValue)

print("Total Profit:", pulp.value(prob.objective))
