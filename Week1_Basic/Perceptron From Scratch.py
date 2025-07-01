'''
What is a Perceptron?
A Perceptron is the simplest type of artificial neuron — the building block of a neural network. It models how a biological neuron works.

Mathematical Equation:
The perceptron computes a weighted sum:

z=w1x1 + w2x2 + ⋯ + wnxn +b = w⋅x + b
Then applies an activation function:
y^ = f(z)
Where:
xi: Input features
wi: Weights associated with inputs
b: Bias (similar to offset)
f: Activation function (like step function, sigmoid, ReLU)

'''
import numpy as np

# Step Function (threshold activation) This function makes the Perceptron a linear binary classifier
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron Class simple single-layer perceptron
class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=10):
        self.weights = np.zeros(input_size)
        self.bias = 0 
        self.lr = lr # Controls the size of weight updates during training
        self.epochs = epochs #How many times to iterate over the training data

    def predict(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return step_function(linear_output)

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                self.weights += self.lr * error * xi #Update weights = weights + learning_rate ⋅ error ⋅ input
                self.bias += self.lr * error #Update bias = bias + learning_rate ⋅ error


X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])  # AND logic

p = Perceptron(input_size=2)
p.train(X, y)

for xi in X:
    print(f"Input: {xi}, Prediction: {p.predict(xi)}")

