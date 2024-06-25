import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from model import FastFPN, YOLOv8Backbone, resize_size


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('_image.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx].replace('_image.jpg', '_mask.jpg'))

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


def split_dataset(dataset, min_chunk_size=100, validation_split=0.2):
    num_images = len(dataset)
    if num_images <= min_chunk_size:
        chunks = [Subset(dataset, range(num_images))]
    else:
        num_chunks = max(1, num_images // min_chunk_size)
        while num_images % num_chunks != 0:
            num_chunks -= 1

        chunk_size = num_images // num_chunks
        chunks = [Subset(dataset, range(i, min(i + chunk_size, num_images))) for i in range(0, num_images, chunk_size)]

    train_chunks = []
    val_chunks = []
    for chunk in chunks:
        indices = list(range(len(chunk)))
        split = int(np.floor(validation_split * len(chunk)))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_chunks.append(Subset(chunk, train_indices))
        val_chunks.append(Subset(chunk, val_indices))

    return train_chunks, val_chunks


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def chunked_training(image_dir, mask_dir, batch_size, num_epochs=30, patience=3):
    best_val_loss = float('inf')
    patience_counter = 0

    image_transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
    ])

    dataset = SegmentationDataset(image_dir, mask_dir, transform=image_transform, mask_transform=mask_transform)
    train_chunks, val_chunks = split_dataset(dataset)

    for chunk_idx, (train_chunk, val_chunk) in enumerate(zip(train_chunks, val_chunks)):
        # 各チャンクごとにモデルをリロード
        backbone = YOLOv8Backbone('yolov8n.pt')
        model = FastFPN()
        pt_file_path = "weight.pt"
        state_dict = torch.load(pt_file_path, map_location=torch.device('cpu'))

        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('model.', '')
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=False)
        model.backbone = backbone

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model = model.float()

        criterion = DiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        train_loader = DataLoader(train_chunk, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_chunk, batch_size=batch_size, shuffle=False, num_workers=4)

        # チャンクの最初にモデルを保存
        torch.save(model.state_dict(), f'weight_chunk_{chunk_idx}_initial.pt')

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0

            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(device).float()
                masks = masks.to(device).float()

                optimizer.zero_grad()

                detections, pred_masks = model(images)

                pred_masks = pred_masks.view(masks.size(0), -1, resize_size, resize_size)
                masks = masks.expand(masks.size(0), pred_masks.size(1), resize_size, resize_size)

                loss = criterion(pred_masks, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                print(f"Chunk {chunk_idx + 1}/{len(train_chunks)}, Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item()}")
                sys.stdout.flush()

            print(f"Chunk {chunk_idx + 1}/{len(train_chunks)}, Epoch {epoch + 1}/{num_epochs}, Average Train Loss: {train_loss / len(train_loader)}")
            sys.stdout.flush()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_idx, (images, masks) in enumerate(val_loader):
                    images = images.to(device).float()
                    masks = masks.to(device).float()

                    detections, pred_masks = model(images)

                    pred_masks = pred_masks.view(masks.size(0), -1, resize_size, resize_size)
                    masks = masks.expand(masks.size(0), pred_masks.size(1), resize_size, resize_size)

                    loss = criterion(pred_masks, masks)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Chunk {chunk_idx + 1}/{len(train_chunks)}, Epoch {epoch + 1}/{num_epochs}, Average Val Loss: {avg_val_loss}")
            sys.stdout.flush()

            scheduler.step()

            # 改善された場合にのみモデルを保存
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'weight_chunk_{chunk_idx}_epoch_{epoch}.pt')
                if os.path.exists('weight.pt'):
                    if os.path.exists('old_weight.pt'):
                        os.remove('old_weight.pt')
                    os.rename('weight.pt', 'old_weight.pt')
                torch.save(model.state_dict(), 'weight.pt')
                print(f"Save model for epoch {epoch + 1}, Val loss: {avg_val_loss}", flush=True)
                sys.stdout.flush()
            else:
                patience_counter += 1
                print(f"No improvement. Epoch: {epoch + 1}, Patience counter: {patience_counter}", flush=True)
                sys.stdout.flush()

            if chunk_idx == len(train_chunks) - 1 and patience_counter >= patience:
                print(f"It doesn't improve, so it stops early at epoch {epoch + 1} of the last chunk.", flush=True)
                sys.stdout.flush()
                break


def start(home_path, batch_size=16):

    image_dir = os.path.join(home_path, "image")
    mask_dir = os.path.join(home_path, "mask")

    chunked_training(image_dir, mask_dir, batch_size)


    shutil.rmtree(home_path)
    
