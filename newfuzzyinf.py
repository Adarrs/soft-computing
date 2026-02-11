import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- 1. Define Antecedents (Inputs) and Consequent (Output) ---
# Service range: 0 to 10
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
# Food range: 0 to 10
food = ctrl.Antecedent(np.arange(0, 11, 1), 'food')
# Tip range: 0 to 25%
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

# --- 2. Define Membership Functions ---
# Service categories
service['poor'] = fuzz.trimf(service.universe, [0, 0, 5])
service['acceptable'] = fuzz.trimf(service.universe, [0, 5, 10])
service['excellent'] = fuzz.trimf(service.universe, [5, 10, 10])

# Food categories
food['bad'] = fuzz.trapmf(food.universe, [0, 0, 1, 3])
food['decent'] = fuzz.trimf(food.universe, [1, 5, 9])
food['great'] = fuzz.trapmf(food.universe, [7, 9, 10, 10])

# Tip categories
tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

# --- 3. Define Fuzzy Rules ---
# Rule 1: If service is poor OR food is bad, then tip is low
rule1 = ctrl.Rule(service['poor'] | food['bad'], tip['low'])

# Rule 2: If service is acceptable AND food is decent, then tip is medium
rule2 = ctrl.Rule(service['acceptable'] & food['decent'], tip['medium'])

# Rule 3: If service is excellent AND food is great, then tip is high
rule3 = ctrl.Rule(service['excellent'] & food['great'], tip['high'])

# Rule 4: If food is decent AND service is poor, then tip is medium (Example of specific rule)
rule4 = ctrl.Rule(food['decent'] & service['poor'], tip['medium'])

# --- 4. Create Control System and Simulation ---
tip_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
tipping = ctrl.ControlSystemSimulation(tip_control)

# --- 5. Compute Output for Specific Inputs ---
print("--- Example 1 ---")
tipping.input['service'] = 6.5
tipping.input['food'] = 9.8

# Perform the fuzzy inference
tipping.compute()

print(f"Service Rating: 6.5/10")
print(f"Food Rating: 9.8/10")
print(f"Recommended Tip: {tipping.output['tip']:.2f}%")

# Visualize the result (Tip membership functions with the centroid line)
tip.view(sim=tipping)
plt.title("Fuzzy Inference Result (Tip)")
plt.show()

# --- 6. Another Example ---
print("\n--- Example 2 ---")
tipping.input['service'] = 2
tipping.input['food'] = 3
tipping.compute()
print(f"Service Rating: 2/10")
print(f"Food Rating: 3/10")
print(f"Recommended Tip: {tipping.output['tip']:.2f}%")
