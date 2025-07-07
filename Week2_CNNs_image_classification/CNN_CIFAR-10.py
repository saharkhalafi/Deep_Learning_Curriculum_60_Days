import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. Transformations for CIFAR-10 images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(([0.5, 0.5, 0.5]), ([0.5, 0.5, 0.5]))  # Normalize RGB channels
])

# 3. Load CIFAR-10 dataset
batch_size = 64
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform) #Image shape: torch.Size([3, 32, 32])
test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

# 4. Define a CNN model
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))                    # [batch, 16, 32, 32]
        x = self.pool(torch.relu(self.conv2(x)))         # [batch, 32, 16, 16]
        x = self.pool(x)                                 # [batch, 32, 8, 8]
        x = self.dropout(x)
        x = x.view(-1, 32 * 8 * 8)                        # flatten: [batch, 2048]
        x = torch.relu(self.fc1(x))                      # [batch, 128]
        return self.fc2(x)                               # [batch, 10]

# 5. Instantiate model, loss, optimizer
model = CIFAR10CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

# 7. Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"\nTest Accuracy: {100 * correct / total:.2f}%")
