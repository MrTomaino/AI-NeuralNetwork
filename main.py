import numpy as np
import sklearn
import sklearn.datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate the dataset
def generate_data(n_samples=500, noise=0.2):
    X, y = sklearn.datasets.make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize parameters for multiple layers
def initialize_parameters(layer_sizes):
    np.random.seed(1)
    parameters = {}
    for l in range(1, len(layer_sizes)):
        parameters[f'W{l}'] = np.random.randn(layer_sizes[l-1], layer_sizes[l]) * 0.01
        parameters[f'b{l}'] = np.zeros((1, layer_sizes[l]))
    return parameters

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# Forward propagation for multiple layers
def forward_propagation(X, parameters):
    activations = {"A0": X}
    L = len(parameters) // 2  # Number of layers
    
    for l in range(1, L):
        Z = np.dot(activations[f'A{l-1}'], parameters[f'W{l}']) + parameters[f'b{l}']
        A = relu(Z)
        activations[f'Z{l}'] = Z
        activations[f'A{l}'] = A
    
    # Output layer with sigmoid activation
    ZL = np.dot(activations[f'A{L-1}'], parameters[f'W{L}']) + parameters[f'b{L}']
    AL = sigmoid(ZL)
    activations[f'Z{L}'] = ZL
    activations[f'A{L}'] = AL
    
    return activations

# Loss calculation
def compute_loss(AL, y):
    m = y.shape[0]
    return -np.sum(y * np.log(AL) + (1 - y) * np.log(1 - AL)) / m

# Backward propagation for multiple layers
def backward_propagation(y, parameters, activations):
    gradients = {}
    L = len(parameters) // 2
    m = y.shape[0]
    y = y.reshape(-1, 1)
    
    # Output layer gradients
    dZL = activations[f'A{L}'] - y
    gradients[f'dW{L}'] = np.dot(activations[f'A{L-1}'].T, dZL) / m
    gradients[f'db{L}'] = np.sum(dZL, axis=0, keepdims=True) / m
    
    # Backpropagate through hidden layers
    for l in range(L-1, 0, -1):
        dA = np.dot(dZL, parameters[f'W{l+1}'].T)
        dZ = dA * (activations[f'Z{l}'] > 0)  # ReLU derivative
        gradients[f'dW{l}'] = np.dot(activations[f'A{l-1}'].T, dZ) / m
        gradients[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True) / m
        dZL = dZ

    return gradients

# Update parameters for multiple layers
def update_parameters(parameters, gradients, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f'W{l}'] -= learning_rate * gradients[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * gradients[f'db{l}']
    return parameters

# Training function
def train_neural_network(X_train, y_train, layer_sizes, learning_rate=0.01, iterations=1000):
    parameters = initialize_parameters(layer_sizes)

    for i in range(iterations):
        activations = forward_propagation(X_train, parameters)
        loss = compute_loss(activations[f'A{len(layer_sizes)-1}'], y_train)
        gradients = backward_propagation(y_train, parameters, activations)
        parameters = update_parameters(parameters, gradients, learning_rate)

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")

    return parameters

# Prediction function
def predict(X, parameters):
    activations = forward_propagation(X, parameters)
    AL = activations[f'A{len(parameters)//2}']
    return (AL > 0.5).astype(int)

# Plotting function
def plot_decision_boundary(X, y, parameters):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = predict(np.c_[xx.ravel(), yy.ravel()], parameters)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title("Decision Boundary")
    plt.show()

# Main function
def main():
    # Step 1: Generate and split the data
    X_train, X_test, y_train, y_test = generate_data(n_samples=500, noise=0.2)

    # Step 2: Define the neural network architecture
    layer_sizes = [2, 5, 5, 5, 3, 1]  # 2 input features, two hidden layers with 3 and 5 units, and 1 output layer
    learning_rate = 0.01        # Learning rate
    iterations = 10000           # Number of iterations

    # Step 3: Train the neural network
    parameters = train_neural_network(X_train, y_train, layer_sizes, learning_rate, iterations)

    # Step 4: Evaluate the model
    train_accuracy = np.mean(predict(X_train, parameters) == y_train.reshape(-1, 1)) * 100
    test_accuracy = np.mean(predict(X_test, parameters) == y_test.reshape(-1, 1)) * 100

    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Testing Accuracy: {test_accuracy:.2f}%")

    # Step 5: Plot the decision boundary
    plot_decision_boundary(X_train, y_train, parameters)

# Entry point
if __name__ == "__main__":
    main()
