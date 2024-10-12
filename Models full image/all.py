import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from collections import Counter

def load_images(base_path, categories):
    data = []
    for idx, category in enumerate(categories):
        category_path = os.path.join(base_path, category)
        for root, _, files in os.walk(category_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(root, file)
                    data.append((file_path, idx))
    return data

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        file_path, label = self.image_paths[idx]
        with Image.open(file_path) as img:
            if self.transform:
                img = self.transform(img)
            return img, label

def get_data_loaders(base_path, categories, batch_size=16):
    data = load_images(base_path, categories)
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])
    
    # Enhanced data augmentation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomErasing(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    train_dataset = ImageDataset(train_data, transform)
    val_dataset = ImageDataset(val_data, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_{model.__class__.__name__.lower()}.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    return avg_loss, accuracy

def main():
    base_path = "C:/Users/PC/OneDrive/Escritorio/VPH/XMAYUS/OneDrive_2024-09-18/Proyecto vph/Categorized Cells"
    categories = ["Normal Cells", "Pre-cancerous Lesions", "Reactive or Reparative Changes"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Loaders
    train_loader, val_loader = get_data_loaders(base_path, categories)
    
    # Models with different learning rates and configurations
    model_configs = {
        "EfficientNet": (models.efficientnet_b7(weights='DEFAULT'), 0.001),
        "DenseNet": (models.densenet201(weights='DEFAULT'), 0.0001),
        "ResNet": (models.resnet50(weights='DEFAULT'), 0.0001)
    }

    best_models = {}
    
    for name, (model, lr) in model_configs.items():
        num_features = model.classifier.in_features if hasattr(model, 'classifier') else model.fc.in_features
        model.classifier = nn.Linear(num_features, len(categories)) if hasattr(model, 'classifier') else nn.Linear(num_features, len(categories))
        
        # Introduce Dropout
        if hasattr(model, 'classifier'):
            model.classifier = nn.Sequential(
                model.classifier,
                nn.Dropout(p=0.5)  # Add dropout with a 50% probability
            )

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        print(f"\nTraining {name}...")
        train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=15)

        best_models[name] = f'best_{name.lower()}.pth'

    print("\nTraining complete. Best models saved:")
    for name, path in best_models.items():
        print(f"{name}: {path}")

if __name__ == "__main__":
    main()
