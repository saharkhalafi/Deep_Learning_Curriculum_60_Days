import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: Load and preprocess MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# Step 2: Define CNN (Corrected Version)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 1 input channel, 32 filters, 3x3 kernel # 1 input channel, 32 filters, 3x3 kernel  # (in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)          # 2x2 pooling
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # 64*5*5=1600
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # [64,1,28,28] → [64,32,26,26]
        x = self.pool(x)               #[64,32,26,26] → [64,32,13,13] ,Output size = (Input_size - Kernel_size + 2*Padding)/Stride + 1 (28 - 3 + 0)/1 + 1 = 26 ,Output_size = Input_size / Kernel_size 26 / 2 = 13
        x = torch.relu(self.conv2(x))  # [64,32,13,13] → [64,64,11,11]
        x = self.pool(x)               # [64,64,11,11] → [64,64,5,5]
        x = x.view(-1, 64 * 5 * 5)     # Flatten to [64,1600]
        x = torch.relu(self.fc1(x))    # [64,1600] → [64,128]
        x = self.fc2(x)                # [64,128] → [64,10]
        return x

# Step 3: Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Step 4: Evaluate
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")



'''

Layer	  Output Shape
Input	  [N, 1, 28, 28]
Conv1	  [N, 32, 26, 26]
MaxPool1  [N, 32, 13, 13]
Conv2	  [N, 64, 11, 11]
MaxPool2  [N, 64, 5, 5]
Flatten	  [N, 1600]
fc1	      [N, 128]
fc2	      [N, 10]

'''
