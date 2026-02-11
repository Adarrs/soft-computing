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
