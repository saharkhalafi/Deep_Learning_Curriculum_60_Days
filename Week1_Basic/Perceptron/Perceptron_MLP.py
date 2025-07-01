import torch
import torch.nn as nn
import torch.optim as optim

# XOR Data
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=torch.float32)
y = torch.tensor([[0.],[1.],[1.],[0.]], dtype=torch.float32)

# Simple 2-layer MLP
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)         # 2 inputs → 4 hidden
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)         # 4 hidden → 1 output

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))     # sigmoid for binary output
        return x

# Model, Loss, Optimizer
model = Net()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training
for epoch in range(5000):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test
with torch.no_grad():
    output = model(X)
    print("\nPredictions:")
    print(torch.round(output))



'''
Concept	               What It Is
Perceptron	           Basic neuron using a weighted sum and step function
MLP (NN)	             Stack of layers (input → hidden → output)
Activation	           Adds non-linearity: ReLU, Sigmoid, Tanh, etc.
Loss Function	         Measures prediction error (e.g., MSE, BCE)
Optimizer	             Updates weights to minimize loss (SGD, Adam…)
Forward Pass	         Computes outputs using current weights
Backpropagation	       Finds gradients to reduce loss via chain rule

'''
