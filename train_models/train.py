import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.notebook import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset():
        # Define transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # C:\Users\hp\Eye-Retina-Degeneration-Detection\dataset\train
    
    
    # Define dataset paths
    train_path = "C:/Users/hp/Eye-Retina-Degeneration-Detection/dataset/train"
    valid_path = "C:/Users/hp/Eye-Retina-Degeneration-Detection/dataset/valid"
    test_path = "C:/Users/hp/Eye-Retina-Degeneration-Detection/dataset/test"
    
    # Define datasets
    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    valid_dataset = datasets.ImageFolder(valid_path, transform=transform)
    test_dataset = datasets.ImageFolder(test_path, transform=transform)
    
    # Define dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader
    
class TransferNet(nn.Module):
    def __init__(self, num_classes=2):
        super(TransferNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Add custom fully connected layer
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_model(train_loader, valid_loader, num_epochs, optimizer_index, learning_rate):
    model = TransferNet()
    model = model.to(device)
    opt = [optim.Adam(model.parameters()), optim.RMSprop(model.parameters()), optim.Adagrad(model.parameters()), optim.SGD(model.parameters())]
    optimizer = opt[optimizer_index]
    optimizer.lr = learning_rate
    criterion = nn.CrossEntropyLoss()

    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []  # List to store training losses
    val_losses = []  # List to store validation losses
    
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(valid_loader.dataset)
        accuracy = correct_preds.double() / total_samples

        scheduler.step(avg_val_loss)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Training Loss: {avg_train_loss:.4f}, '
              f'Validation Loss: {avg_val_loss:.4f}, '
              f'Accuracy: {accuracy:.4f}')
        
    # Plotting the training and validation loss
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    # plt.legend()
    # plt.show()
    
    # Convert the plot to an image
    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    
    torch.save(model.state_dict(),  "C:/Users/hp/Eye-Retina-Degeneration-Detection/models/transfer_model.pth")
    return image

def evaluate_model(test_loader):
    
    model = TransferNet()
    model.load_state_dict(torch.load("C:/Users/hp/Eye-Retina-Degeneration-Detection/models/transfer_model.pth",map_location=device)
)
    model.to(device)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    correct = sum([1 if pred == label else 0 for pred, label in zip(all_preds, all_labels)])
    total = len(all_labels)
    accuracy = correct / total
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, cm

def confusion_matrixes(cm):
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Convert the plot to an image
    fig = plt.gcf()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return image
    
def run(epoches, optimizer_index, learning_rate):
    
    train_loader, valid_loader, test_loader = load_dataset()
    loss_image = train_model(train_loader, valid_loader, epoches, optimizer_index, learning_rate)
    acc, cm = evaluate_model(test_loader)
    con_mat = confusion_matrixes(cm)
    
    visuals = [con_mat, loss_image]
    
    return acc, visuals
    
if __name__ == "__main__":
    run(1, 1, 0.01)




