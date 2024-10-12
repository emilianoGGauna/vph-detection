import os
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

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

    # Check for class balance
    labels = [label for _, label in data]
    print(f"Class distribution: {Counter(labels)}")

    # Stratified split into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, stratify=labels, random_state=42)
    
    # Image transformations with data augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet50 requires input size of 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalization for ResNet50
    ])
    
    train_dataset = ImageDataset(train_data, transform)
    val_dataset = ImageDataset(val_data, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_and_evaluate(base_path, categories, num_epochs=10, batch_size=16):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_loader, val_loader = get_data_loaders(base_path, categories, batch_size)
    
    # Initialize ResNet50 with updated weights parameter
    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_features = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_features, len(categories))
    resnet50 = resnet50.to(device)

    # Unfreeze more layers for fine-tuning
    for param in resnet50.parameters():
        param.requires_grad = False
    for param in list(resnet50.children())[-4:]:
        for p in param.parameters():
            p.requires_grad = True

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet50.parameters()), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    for epoch in range(num_epochs):
        # Training phase
        resnet50.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet50(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(resnet50.parameters(), max_norm=2.0)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        val_loss, val_accuracy = evaluate_model(resnet50, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(resnet50.state_dict(), 'best_resnet_model.pth')
            patience_counter = 0  # Reset the counter if we get a better model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    
    print("Training complete.")

    # Load best model and final evaluation
    resnet50.load_state_dict(torch.load('best_resnet_model.pth'))
    evaluate_model(resnet50, val_loader, criterion, device, final_report=True, categories=categories)

def evaluate_model(model, dataloader, criterion, device, final_report=False, categories=None):
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

    if final_report and categories:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=categories))

    return avg_loss, accuracy

def main():
    base_path = "C:/Users/PC/OneDrive/Escritorio/VPH/XMAYUS/OneDrive_2024-09-18/Proyecto vph/Categorized Cells"
    categories = ["Normal Cells", "Pre-cancerous Lesions", "Reactive or Reparative Changes"]
    
    train_and_evaluate(base_path, categories, num_epochs=10, batch_size=16)

if __name__ == "__main__":
    main()
