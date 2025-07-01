import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms #Datasets and transforms (image processing)
from torch.utils.data import DataLoader #Efficient loading of batches of data

# 1. Transform: Convert PIL image to tensor and normalize it
transform = transforms.Compose([
    transforms.ToTensor(), 
  #Converts image of shape (H, W, C) to (C, H, W) and normalizes pixel values (0–255) to (0–1) | PyTorch
   (C, H, W) — channels first
  
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std of MNIST | Normalize(mean, std)
])

# 2. Download the dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# 3. Define a simple MLP network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Input layer, The MNIST image is 28×28 pixels = 784 inputs, Fully Connected Layer 1
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)     # Output layer for 10 digits 128 → 10 (each output = score for a digit class 0–9)

    def forward(self, x):
        x = x.view(-1, 28*28)   # Flatten the image , Flatten the 2D image tensor (batch_size, 1, 28, 28) ⇒ (batch_size, 784)
        x = self.relu(self.fc1(x)) # Hidden layer + ReLU
        x = self.fc2(x)  # Output layer
        return x

# 4. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss() # Loss function → CrossEntropyLoss combines LogSoftmax + NLLLoss (so raw logits are fine)
optimizer = optim.Adam(model.parameters(), lr=0.001) #Adam (adaptive learning algorithm), good default for small networks

# 5. Training Loop
for epoch in range(5):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}")

# 6. Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
