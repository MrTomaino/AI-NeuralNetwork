import numpy as np
import sklearn
import sklearn.datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate the dataset
def generate_data(n_samples=500, noise=0.2):
    X, y = sklearn.datasets.make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize parameters
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(1)
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

# Loss calculation
def compute_loss(A2, y):
    m = y.shape[0]
    return -np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2)) / m

# Backward propagation
def backward_propagation(X, y, Z1, A1, Z2, A2, W1, W2):
    m = X.shape[0]
    dZ2 = A2 - y.reshape(-1, 1)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dZ1 = np.dot(dZ2, W2.T) * (Z1 > 0)  # Derivative of ReLU
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

# Update parameters
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Training function
def train_neural_network(X_train, y_train, hidden_size=3, learning_rate=0.01, iterations=1000):
    input_size = X_train.shape[1]
    output_size = 1

    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X_train, W1, b1, W2, b2)
        loss = compute_loss(A2, y_train)
        dW1, db1, dW2, db2 = backward_propagation(X_train, y_train, Z1, A1, Z2, A2, W1, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")

    return W1, b1, W2, b2

# Prediction function
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return (A2 > 0.5).astype(int)

# Plotting function
def plot_decision_boundary(X, y, W1, b1, W2, b2):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = predict(np.c_[xx.ravel(), yy.ravel()], W1, b1, W2, b2)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title("Decision Boundary")
    plt.show()

# Main function
def main():
    # Step 1: Generate and split the data
    X_train, X_test, y_train, y_test = generate_data(n_samples=500, noise=0.2)

    # Step 2: Train the neural network
    hidden_size = 3         # Students can change this value
    learning_rate = 0.01    # Students can change this value
    iterations = 1000       # Students can change this value

    W1, b1, W2, b2 = train_neural_network(X_train, y_train, hidden_size, learning_rate, iterations)

    # Step 3: Evaluate the model
    train_accuracy = np.mean(predict(X_train, W1, b1, W2, b2) == y_train.reshape(-1, 1)) * 100
    test_accuracy = np.mean(predict(X_test, W1, b1, W2, b2) == y_test.reshape(-1, 1)) * 100

    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Testing Accuracy: {test_accuracy:.2f}%")

    # Step 4: Plot the decision boundary
    plot_decision_boundary(X_train, y_train, W1, b1, W2, b2)

# Entry point
if __name__ == "__main__":
    main()
