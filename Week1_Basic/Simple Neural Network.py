''' 
Deep Learning is a subfield of Machine Learning (ML) that uses algorithms inspired by the structure and function of the brain’s neural networks. It involves training deep neural networks — networks with many layers — to learn patterns from large amounts of data, especially high-dimensional data like images, text, speech, and more.
👉 Deep Learning = Feature Learning + Neural Networks + Lots of Data + Big Compute
🧠 Biological Inspiration:
Deep Learning is loosely inspired by how the human brain works. Specifically:
•	Neuron: Basic unit that processes input and gives an output
•	Artificial Neuron (Perceptron): Mathematical function mimicking neurons
•	Neural Network: Collection of layered neurons
Each layer transforms input data to learn higher-level features — the deeper the layers, the more abstract they become.
🏗️ Structure of a Deep Neural Network:
A typical neural network consists of:
1.	Input Layer: Raw data (pixels, text tokens, etc.)
2.	Hidden Layers: Each layer transforms its input into a more useful representation
3.	Output Layer: Final prediction (class, text, number)
Each layer has parameters (weights and biases) which are updated during training using backpropagation and gradient descent.
💡 Why Deep Learning Is Powerful:
•	Learns features directly from raw data
•	Can model complex functions with enough layers
•	Great for unstructured data (images, texts)
•	Benefits massively from GPUs/TPUs
•	Scales well with large datasets
'''

import torch
import torch.nn as nn
import torch.optim as optim

# Sample Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 16)   # Input layer: 10 features → 16 neurons
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)    # Output layer (binary)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# Dummy data
inputs = torch.randn(5, 10)         # 5 samples, each with 10 features
labels = torch.randint(0, 2, (5, 1)).float()

# Model + Loss + Optimizer
model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop (1 Epoch)
for epoch in range(1):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item():.4f}")
