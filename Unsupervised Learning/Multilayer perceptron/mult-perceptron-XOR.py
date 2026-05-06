import numpy as np
import random

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derive of activation function
def sigmoid_derivative(x):
    return x * (1 - x)


# Define the entries and expected values
def load_data():
    # Entries for XOR
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    # Expected output
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    return X, y


# Main logic for multilayer perceptron
def train_and_evaluate(X, y, hidden_neurons, max_epochs, learning_rate=0.1):
    input_neurons = X.shape[1]
    output_neurons = y.shape[1]

    # Initialize weights with numbers between -1, 1
    W_hidden = np.array([[random.uniform(-1, 1) for _ in range(hidden_neurons)] for _ in range(input_neurons)])
    W_output = np.array([[random.uniform(-1, 1) for _ in range(output_neurons)] for _ in range(hidden_neurons)])

    # Initialize bias (1)
    b_hidden = np.ones((1, hidden_neurons))
    b_output = np.ones((1, output_neurons))

    for epoch in range(max_epochs):
        # Forward propagation
        # Point product, weighted sum, and activation function
        hidden_layer_input = np.dot(X, W_hidden) + b_hidden
        hidden_layer_activation = sigmoid(hidden_layer_input)

        # Same calculations as before
        output_layer_input = np.dot(hidden_layer_activation, W_output) + b_output
        predicted_output = sigmoid(output_layer_input)

        # Calculate the error
        error = y - predicted_output
        mean_abs_error = np.mean(np.abs(error))

        # Shows the error and result variations
        current_results = np.round(predicted_output.flatten(), 3)
        print(f"Epoch {epoch} - Error: {mean_abs_error:.6f}\nActual results: {current_results}")

        # Stop condition: when epoch = max_epochs or when the rounded predictions are
        # equal to the real values and the error is less than 0.05
        predictions_rounded = np.round(predicted_output)
        if np.array_equal(predictions_rounded, y) and mean_abs_error < 0.05:
            current_results = np.round(predicted_output.flatten(), 3)
            print(f"Epoch {epoch} - Error: {mean_abs_error:.6f}\nActual results: {current_results}")
            print(f"\nStop condition reached at epoch {epoch}.")
            break

        # Back propagation
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        error_hidden_layer = d_predicted_output.dot(W_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)

        # Weights and bias actualization for L elements
        W_output += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
        b_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate

        # Weights and bias actualization for L-1 elements
        W_hidden += X.T.dot(d_hidden_layer) * learning_rate
        b_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Final results
    print("\n\nFinal results:\n")
    print("Input X:")
    print(X)
    print("\nReal output Y:")
    print(y)
    print("\nRaw output:")
    print(predicted_output)
    print("\nRounded output:")
    print(np.round(predicted_output))


def main():
    max_epochs = 10000
    hidden_neurons = 9

    X, y = load_data()

    train_and_evaluate(X, y, hidden_neurons, max_epochs)


if __name__ == "__main__":
    main()