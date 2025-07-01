import numpy as np

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): #Used during backpropagation to calculate the gradients
    return x * (1 - x)

# Sample dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Initialize weights
np.random.seed(42)
input_neurons = 2
hidden_neurons = 4
output_neurons = 1

#Random Initialization
W1 = np.random.randn(input_neurons, hidden_neurons) #weights are initialized randomly to break symmetry
b1 = np.zeros((1, hidden_neurons)) #Biases are initialized to zeros
W2 = np.random.randn(hidden_neurons, output_neurons)
b2 = np.zeros((1, output_neurons))

# Training loop
lr = 0.1
epochs = 5000

for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    output = sigmoid(final_input)

    # Compute error and loss
    loss = np.mean((y - output)**2)

    # Backpropagation
    error = y - output
    d_output = error * sigmoid_derivative(output)

    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update Weights and Biases (Gradient Descent)
    '''
    Weight updates = learning_rate × gradient
    Use matrix multiplication to compute updates efficiently
    Bias updates = sum of gradients
    '''
    W2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Predictions
print("\nFinal Predictions:")
print(np.round(output))


'''
Step	                 Purpose
1. Activation            Functions	Introduce non-linearity
2. Dataset	             Sample: not linearly separable
3. Initialization	       Random weights to break symmetry
4. Forward Pass	         Compute activations of hidden and output layers
5. Loss	                 Measure how bad predictions are
6. Backpropagation	     Calculate gradients of weights to minimize loss
7. Weight Updates	       Apply gradient descent
8. Training Loop	       Repeat to learn from data
9. Final Output	         Predict using learned weights


'''
