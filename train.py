'''
    I referred this code based on the github implementation: https://github.com/kuan-wang/pytorch-mobilenet-v3
    My github profile link: https://github.com/ejmfrancisco

'''
# Assuming the MobileNetV3 class and necessary imports are defined as in your initial code snippet
import os
import torch
import json
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from model.mobilenetv3_pytorch import MobileNetV3
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image

# Define paths
dataset_path = "<Your dataset path.>"

# Define transformations with data augmentation for training and standard transforms for validation/test
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset = ImageFolder(root=dataset_path, transform=train_transforms)
class_to_idx = dataset.class_to_idx

# Save class_to_idx to a JSON file
with open('class_to_idx.json', 'w') as json_file:
    json.dump(class_to_idx, json_file)

test_dataset = ImageFolder(root=dataset_path, transform=test_transforms)

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, test_size])
_, _, test_dataset = random_split(test_dataset, [train_size, val_size, test_size])  # Apply test transform

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Instantiate the model
num_classes = len(dataset.classes)
model = MobileNetV3(config_name="large", classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3, verbose=True)

# Training with early stopping and checkpointing
best_val_loss = float("inf")
early_stopping_patience = 10
epochs_without_improvement = 0
epochs = 30


for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    
    # Validate
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Early stopping and checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model_60epochs.pth")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Stopping early at epoch {epoch+1}")
            break
    scheduler.step(val_loss)

# Load the best model
model.load_state_dict(torch.load("best_model_60epochs.pth"))

# Test the model
model.eval()
test_accuracy = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_accuracy += (predicted == labels).sum().item()
test_accuracy = 100 * test_accuracy / len(test_dataset)
print(f"Test Accuracy: {test_accuracy:.2f}%")
