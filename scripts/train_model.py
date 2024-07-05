import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class TeethSegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

def load_data(data_dir):
    images = np.load(os.path.join(data_dir, 'train_images.npy'))
    masks = np.load(os.path.join(data_dir, 'train_masks.npy'))
    return images, masks

def train_model(data_dir, epochs=10, batch_size=4, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, masks = load_data(data_dir)

    dataset = TeethSegmentationDataset(images, masks, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet(num_classes=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")

    print("Training completed.")
    torch.save(model.state_dict(), os.path.join(data_dir, 'model.pth'))

if __name__ == "__main__":
    data_dir = '../data/processed'
    train_model(data_dir)
