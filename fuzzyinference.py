import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Antecedents and Consequent
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
food = ctrl.Antecedent(np.arange(0, 11, 1), 'food')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

# Membership functions
service['poor'] = fuzz.trimf(service.universe, [0, 0, 5])
service['acceptable'] = fuzz.trimf(service.universe, [0, 5, 10])
service['excellent'] = fuzz.trimf(service.universe, [5, 10, 10])

food['bad'] = fuzz.trapmf(food.universe, [0, 0, 1, 3])
food['decent'] = fuzz.trimf(food.universe, [1, 5, 9])
food['great'] = fuzz.trapmf(food.universe, [7, 9, 10, 10])

tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

# Rules
rule1 = ctrl.Rule(service['poor'] | food['bad'], tip['low'])
rule2 = ctrl.Rule(service['acceptable'] & food['decent'], tip['medium'])
rule3 = ctrl.Rule(service['excellent'] & food['great'], tip['high'])
rule4 = ctrl.Rule(food['decent'] & service['poor'], tip['medium'])

# Control system
tip_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
tipping = ctrl.ControlSystemSimulation(tip_control)

# Inputs
tipping.input['service'] = 6.5
tipping.input['food'] = 9.8

# Compute
tipping.compute()

print(f"Service Rating: 6.5/10")
print(f"Food Rating: 9.8/10")
print(f"Recommended Tip: {tipping.output['tip']:.2f}%")

# Visualize
tip.view(sim=tipping)
plt.show()

# Example 2
print("\n--- Example 2 ---")
tipping.input['service'] = 2
tipping.input['food'] = 3
tipping.compute()
print(f"Recommended Tip: {tipping.output['tip']:.2f}%")
