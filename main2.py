# %%
import os
import cv2
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as maskUtils
from tqdm import tqdm
import time

# %%
class RoofDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.file_list = self._get_file_list()
        self.transform = transform

    def _get_file_list(self):
        file_list = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.json'):
                    file_list.append(os.path.join(root, file))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        json_path = self.file_list[idx]
        with open(json_path, 'r') as f:
            data = json.load(f)
            image_path = os.path.join(os.path.dirname(json_path), data['image']['file_name'])
            image = cv2.imread(image_path)
            image = image / 255.0  # Normalize the image

            mask = np.zeros((data['image']['height'], data['image']['width']), dtype=np.uint8)
            for annotation in data['annotations']:
                rle = annotation['segmentation']
                binary_mask = maskUtils.decode(rle)
                mask = np.maximum(mask, binary_mask)
            mask = mask / 255.0  # Normalize the mask

            # Transpose the image and mask to match the expected shape [channels, height, width]
            image = np.transpose(image, (2, 0, 1))
            mask = np.expand_dims(mask, axis=0)

            image = torch.tensor(image, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)

            return image, mask

# Example usage
train_data_dir = './roof_data/train'
val_data_dir = './roof_data/valid'

train_dataset = RoofDataset(train_data_dir)
val_dataset = RoofDataset(val_data_dir)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.center = self.conv_block(512, 1024)
        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        center = self.center(F.max_pool2d(enc4, 2))
        dec4 = self.decoder4(torch.cat([F.interpolate(center, scale_factor=2, mode='bilinear', align_corners=True), enc4], 1))
        dec3 = self.decoder3(torch.cat([F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True), enc3], 1))
        dec2 = self.decoder2(torch.cat([F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), enc2], 1))
        dec1 = self.decoder1(torch.cat([F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), enc1], 1))
        return torch.sigmoid(self.final(dec1))

# Example usage
model = UNet()

# %%
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)

# Wrap the model with DataParallel to use multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 50
best_val_loss = float('inf')

# Create the folder to save model checkpoints
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    train_loss = 0.0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        for images, masks in train_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            pbar.update(1)

    train_loss /= len(train_loader.dataset)
    epoch_time = time.time() - start_time
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Time: {epoch_time:.2f} seconds')

    # Save the model at the end of each epoch
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth'))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item() * images.size(0)

    val_loss /= len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}')

    # Save the best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
        print(f'Saved best model with validation loss: {val_loss:.4f}')


