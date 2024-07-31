import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm  # for efficient loading of EfficientNet

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data paths
train_dir = 'D:\Folder\Med Scan\split\train'  # Update this
val_dir = 'D:\Folder\Med Scan\split\validation'  # Update this

best_model_path = 'D:\Folder\Med Scan\Models\EfficientNetB0\best_efficientnet_model.pth'  # Update this
final_model_path = 'D:\Folder\Med Scan\Models\EfficientNetB0\final_efficientnet_model.pth'  # Update this

# Hyperparameters
num_classes = 29
batch_size = 32
num_epochs = 50
learning_rate = 0.001

# Data transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load EfficientNet-B0 model
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
model = model.to(device)

# Freeze early layers
for name, param in model.named_parameters():
    if 'classifier' not in name:  # Freeze all layers except the classifier
        param.requires_grad = False

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs, best_model_path, final_model_path):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in (train_loader if phase == 'train' else val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_dataset if phase == 'train' else val_dataset)
            epoch_acc = running_corrects.double() / len(train_dataset if phase == 'train' else val_dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_path)

        print()

    print(f'Best val Acc: {best_acc:4f}')
    torch.save(model.state_dict(), final_model_path)
    return model

# Train the model
model = train_model(model, criterion, optimizer, scheduler, num_epochs, best_model_path, final_model_path)
