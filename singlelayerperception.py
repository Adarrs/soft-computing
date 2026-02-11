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
