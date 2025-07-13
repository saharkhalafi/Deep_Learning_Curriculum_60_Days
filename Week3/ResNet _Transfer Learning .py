import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm  # For EfficientNet
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ImageNet mean/std 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize CIFAR-10 for models that expect 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# CIFAR-10 dataset (10 classes)
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64)


model_resnet_transfer = resnet18(pretrained=True)  # Load ImageNet pretrained weights

# Freeze feature extractor
for param in model_resnet_transfer.parameters():
    param.requires_grad = False

# Replace final FC layer for CIFAR-10
model_resnet_transfer.fc = nn.Linear(512, 10)
model_resnet_transfer = model_resnet_transfer.to(device)



def train(model, loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(loader):.4f}")

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
model_resnet_transfer = resnet18(pretrained=True)  # Load ImageNet pretrained weights


model = model_resnet_transfer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
train(model, train_loader, criterion, optimizer)
evaluate(model, test_loader)
