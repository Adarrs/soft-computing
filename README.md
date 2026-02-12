# soft-computing
1.pip install numpy

2.pip install numpy
pip install matplotlib
  pip install scikit-fuzzy

3.pip install numpy
pip install matplotlib
  pip install scikit-fuzzy

4.pip install numpy
pip install matplotlib

5.pip install numpy
pip install matplotlib

6.pip install numpy
pip install matplotlib

1.#Operations
import numpy as np

def fuzzy_union_or(A, B, operator='max'):
    if len(A) != len(B):
        raise ValueError("Fuzzy sets must have same length")

    if operator == 'max':
        return np.maximum(A, B)
    else:
        raise NotImplementedError("Only max operator supported")

def fuzzy_intersection_and(A, B, operator='min'):
    if len(A) != len(B):
        raise ValueError("Fuzzy sets must have same length")

    if operator == 'min':
        return np.minimum(A, B)
    else:
        raise NotImplementedError("Only min operator supported")

def fuzzy_complement_not(A):
    return 1 - A


# Universe of discourse
U = np.array([1, 2, 3, 4, 5])
print(f"Universe of Discourse (U): {U}\n")

# Fuzzy sets
A = np.array([1.0, 0.8, 0.4, 0.1, 0.0])
B = np.array([0.0, 0.1, 0.3, 0.7, 1.0])

print("Original Sets")
print(f"Fuzzy Set A: {A}")
print(f"Fuzzy Set B: {B}\n")

# Fuzzy Union
A_OR_B = fuzzy_union_or(A, B)
print("Fuzzy UNION (A OR B)")
print(A_OR_B, "\n")

# Fuzzy Intersection
A_AND_B = fuzzy_intersection_and(A, B)
print("Fuzzy INTERSECTION (A AND B)")
print(A_AND_B, "\n")

# Fuzzy Complement
NOT_A = fuzzy_complement_not(A)
print("Fuzzy COMPLEMENT (NOT A)")
print(NOT_A, "\n")

NOT_B = fuzzy_complement_not(B)
print("Fuzzy COMPLEMENT (NOT B)")
print(NOT_B)


2.#FuzzyInference

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


3.#defuzzyfication

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


def demonstrate_defuzzification(universe, aggregated_mf):
    """
    Applies and compares five common defuzzification techniques
    to a given aggregated membership function (MF).
    """

    # 1. Centroid (Center of Gravity)
    cog = fuzz.defuzz(universe, aggregated_mf, 'centroid')

    # 2. Bisector (Bisector of Area)
    boa = fuzz.defuzz(universe, aggregated_mf, 'bisector')

    # 3. Mean of Maximum
    mom = fuzz.defuzz(universe, aggregated_mf, 'mom')

    # 4. Smallest of Maximum
    som = fuzz.defuzz(universe, aggregated_mf, 'som')

    # 5. Largest of Maximum
    lom = fuzz.defuzz(universe, aggregated_mf, 'lom')

    # --- Print Results ---
    print("\n--- Defuzzification Results ---")
    print(f"Centroid (CoG): {cog:.4f}")
    print(f"Bisector (BoA): {boa:.4f}")
    print(f"Mean of Maximum (MoM): {mom:.4f}")
    print(f"Smallest of Max (SoM): {som:.4f}")
    print(f"Largest of Max (LoM): {lom:.4f}")

    # --- Plot Visualization ---
    plt.figure(figsize=(10, 6))
    plt.plot(universe, aggregated_mf, 'b', linewidth=2.5, label='Aggregated Fuzzy Set')

    # Draw vertical lines for defuzzified values
    plt.axvline(cog, color='r', linestyle='--', label=f'Centroid ({cog:.2f})')
    plt.axvline(boa, color='g', linestyle='-.', label=f'Bisector ({boa:.2f})')
    plt.axvline(mom, color='k', linestyle=':', label=f'MoM ({mom:.2f})')
    plt.axvline(som, color='c', linestyle=':', label=f'SoM ({som:.2f})')
    plt.axvline(lom, color='m', linestyle=':', label=f'LoM ({lom:.2f})')

    plt.title('Comparison of Defuzzification Techniques')
    plt.ylabel('Membership Degree')
    plt.xlabel('Output Universe')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.show()


# --- Define Universe of Discourse ---
X = np.arange(0, 26, 0.1)

# Example fuzzy membership functions
mf_1 = fuzz.trapmf(X, [0, 5, 8, 11])
mf_2 = fuzz.trimf(X, [9, 15, 25])

# Aggregation (Max operation)
aggregated_mf = np.fmax(mf_1 * 0.7, mf_2 * 1.0)

# Run the function
demonstrate_defuzzification(X, aggregated_mf)

4.#Perceptron

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Activation Function (Step Function) ---
def step_function(weighted_sum):
    """
    Returns 1 if weighted sum >= 0, else 0
    """
    return 1 if weighted_sum >= 0 else 0


# --- 2. Perceptron Class ---
class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1, max_epochs=50):
        # Initialize weights and bias randomly
        self.weights = np.random.uniform(-0.5, 0.5, num_inputs)
        self.bias = np.random.uniform(-0.5, 0.5, 1)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.errors = []

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return step_function(weighted_sum)

    def train(self, training_inputs, labels):
        print(f"--- Training Perceptron (Epochs: {self.max_epochs}, Rate: {self.learning_rate}) ---")

        for epoch in range(self.max_epochs):
            total_error = 0

            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                total_error += abs(error)

                # Update rule
                if error != 0:
                    self.weights += self.learning_rate * error * inputs
                    self.bias += self.learning_rate * error

            self.errors.append(total_error)

            if total_error == 0:
                print(f"✅ Converged successfully at Epoch {epoch + 1}")
                break


            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Total Error: {total_error}")

        print("\n--- Training Complete ---")
        print(f"Final Weights: {self.weights}")
        print(f"Final Bias: {self.bias[0]:.4f}")


# --- 3. AND Gate Training Data ---
X_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_train = np.array([0, 0, 0, 1])


# --- 4. Train the Perceptron ---
perceptron = Perceptron(num_inputs=2, learning_rate=0.1, max_epochs=50)
perceptron.train(X_train, y_train)


# --- 5. Test the Model ---
print("\n--- Testing Model Predictions ---")
for inputs, expected in zip(X_train, y_train):
    prediction = perceptron.predict(inputs)
    status = "Correct" if prediction == expected else "Incorrect"
    print(f"Input: {inputs}, Predicted: {prediction}, Expected: {expected} ({status})")


# --- 6. Plot Error Graph ---
plt.figure(figsize=(8, 4))
plt.plot(perceptron.errors, marker='o')
plt.title("Perceptron Training Error Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Total Misclassifications")
plt.grid(True)
plt.show()

5.#backpropogation

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Activation Function and Derivative (Sigmoid) ---
def sigmoid(x):
    x = np.clip(x, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)


# --- 2. Multilayer Perceptron Class ---
class MLP_Backpropagation:
    def __init__(self, input_size, hidden_size, output_size,
                 learning_rate=0.2, max_epochs=10000):

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        # Initialize weights and biases
        self.W_ih = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
        self.b_h = np.zeros((1, hidden_size))

        self.W_ho = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))
        self.b_o = np.zeros((1, output_size))

        self.errors = []

    def forward_pass(self, X):
        # Hidden layer
        self.net_h = np.dot(X, self.W_ih) + self.b_h
        self.out_h = sigmoid(self.net_h)

        # Output layer
        self.net_o = np.dot(self.out_h, self.W_ho) + self.b_o
        self.out_o = sigmoid(self.net_o)

        return self.out_o

    def backward_pass(self, X, y):
        # Output layer error
        error_o = y - self.out_o
        d_o = error_o * sigmoid_derivative(self.out_o)

        # Hidden layer error
        error_h = d_o.dot(self.W_ho.T)
        d_h = error_h * sigmoid_derivative(self.out_h)

        # Update weights and biases
        self.W_ho += self.out_h.T.dot(d_o) * self.learning_rate
        self.b_o += np.sum(d_o, axis=0, keepdims=True) * self.learning_rate

        self.W_ih += X.T.dot(d_h) * self.learning_rate
        self.b_h += np.sum(d_h, axis=0, keepdims=True) * self.learning_rate

        return np.mean(error_o ** 2)

    def train(self, X_train, y_train):
        print("--- Training MLP for XOR Problem ---")

        for epoch in range(self.max_epochs):
            self.forward_pass(X_train)
            mse = self.backward_pass(X_train, y_train)
            self.errors.append(mse)

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}/{self.max_epochs}, MSE: {mse:.6f}")

        print("\n--- Training Complete ---")


# --- 3. XOR Dataset ---
X_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_train = np.array([
    [0],
    [1],
    [1],
    [0]
])


# --- 4. Create and Train MLP ---
mlp = MLP_Backpropagation(
    input_size=2,
    hidden_size=4,
    output_size=1,
    learning_rate=0.2,
    max_epochs=10000
)

mlp.train(X_train, y_train)


# --- 5. Test the Model ---
print("\n--- Testing Model Predictions ---")
predictions = mlp.forward_pass(X_train)

for inputs, pred, expected in zip(X_train, predictions, y_train):
    predicted_class = 1 if pred[0] >= 0.5 else 0
    status = "Correct" if predicted_class == expected[0] else "Incorrect"
    print(f"Input: {inputs}, Output: {pred[0]:.4f}, "
          f"Predicted: {predicted_class}, Expected: {expected[0]} ({status})")


# --- 6. Plot Training Error ---
plt.figure(figsize=(10, 5))
plt.plot(mlp.errors)
plt.title("MLP Training Error (MSE) Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()

6.# Implementation of Simple Neural Network (McCulloch-Pitts model)

import numpy as np


# --- 1. Activation Function (Threshold Logic) ---
def mcp_activation(net_input, threshold):
    """
    Output = 1 if net_input >= threshold else 0
    """
    return 1 if net_input >= threshold else 0


# --- 2. MCP Neuron Function ---
def mcp_neuron(inputs, weights, threshold):
    """
    net_input = Sum(input_i * weight_i)
    """
    inputs = np.array(inputs)
    weights = np.array(weights)

    net_input = np.dot(inputs, weights)
    output = mcp_activation(net_input, threshold)

    return net_input, output


# --- 3. Logical OR Gate ---
def implement_or_gate():

    print("\n--- Implementing Logical OR Gate ---")

    weights = [1, 1]
    threshold = 1

    test_cases = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1)
    ]

    print(f"Weights: {weights}, Threshold: {threshold}")
    print("Input (A, B) | Net Input | Output | Expected")
    print("-" * 45)

    for inputs, expected in test_cases:
        net_input, output = mcp_neuron(inputs, weights, threshold)
        print(f"{inputs[0]}, {inputs[1]} | {net_input} | {output} | {expected}")


# --- 4. Logical NOT Gate ---
def implement_not_gate():

    print("\n--- Implementing Logical NOT Gate ---")

    weights = [-1]
    threshold = 0

    test_cases = [
        ([0], 1),
        ([1], 0)
    ]

    print(f"Weights: {weights}, Threshold: {threshold}")
    print("Input (A) | Net Input | Output | Expected")
    print("-" * 45)

    for inputs, expected in test_cases:
        net_input, output = mcp_neuron(inputs, weights, threshold)
        print(f"{inputs[0]} | {net_input} | {output} | {expected}")


# --- 5. Run ---
implement_or_gate()
implement_not_gate()


