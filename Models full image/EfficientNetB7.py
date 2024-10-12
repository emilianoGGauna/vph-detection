import os
import time
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

def get_data_loaders(base_path, categories, batch_size=8):
    data = load_images(base_path, categories)
    
    # Image transformations with enhanced data augmentation
    transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomErasing(p=0.5),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    dataset = ImageDataset(data, transform)
    
    # Split into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_and_evaluate(base_path, categories, num_epochs=15, batch_size=8):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_loader, val_loader = get_data_loaders(base_path, categories, batch_size)
    
    # Initialize EfficientNet-B7 with pretrained weights
    efficientnet_b7 = models.efficientnet_b7(weights='DEFAULT')  # Updated line
    num_features = efficientnet_b7.classifier[1].in_features
    efficientnet_b7.classifier[1] = nn.Linear(num_features, len(categories))
    efficientnet_b7 = efficientnet_b7.to(device)

    # Unfreeze more layers for fine-tuning
    for param in efficientnet_b7.features.parameters():
        param.requires_grad = False
    for param in list(efficientnet_b7.features.children())[-4:]:
        for p in param.parameters():
            p.requires_grad = True

    # Define loss function and optimizer
    class_counts = Counter([label for _, label in load_images(base_path, categories)])
    class_weights = [1.0 / class_counts[i] for i in range(len(categories))]
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, efficientnet_b7.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Training loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    for epoch in range(num_epochs):
        # Training phase
        efficientnet_b7.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = efficientnet_b7(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(efficientnet_b7.parameters(), max_norm=2.0)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        val_loss, val_accuracy = evaluate_model(efficientnet_b7, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(efficientnet_b7.state_dict(), 'best_efficientnet_model.pth')
            patience_counter = 0  # Reset the counter if we get a better model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    
    print("Training complete.")

    # Load best model and final evaluation
    efficientnet_b7.load_state_dict(torch.load('best_efficientnet_model.pth'))
    evaluate_model(efficientnet_b7, val_loader, criterion, device, final_report=True, categories=categories)

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
    
    train_and_evaluate(base_path, categories, num_epochs=15, batch_size=16)

if __name__ == "__main__":
    main()
