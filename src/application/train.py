import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from model import FastFPN, YOLOv8Backbone, resize_size
import sys
import shutil


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
        mask = Image.open(mask_path).convert("L")  # マスクをグレースケールに変換

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


def save_model(model, path):
    torch.save(model.state_dict(), path)
    if os.path.exists(path):
        if os.path.exists('old_weight.pt'):
            os.remove('old_weight.pt')
        os.rename(path, 'old_weight.pt')
    torch.save(model.state_dict(), path)


def training(image_dir, mask_dir, batch_size, num_epochs=30, patience=3):
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初回ロード
    backbone = YOLOv8Backbone('yolov8n.pt')
    model = FastFPN()
    pt_file_path = "weight.pt"

    if os.path.exists(pt_file_path):
        state_dict = torch.load(pt_file_path, map_location=torch.device('cpu'))
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('model.', '')
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=False)
        print("weight.ptから既存の重みをロードしました")
    else:
        print("既存の重みが見つからないため、新しいモデルで開始します")

    model.backbone = backbone
    model = model.to(device)
    model = model.float()

    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for chunk_idx, (train_chunk, val_chunk) in enumerate(zip(train_chunks, val_chunks)):
        # デバッグ用のプリント文
        print(f"バッチサイズ (型変換前): {batch_size}", flush=True)
        batch_size = int(batch_size)  # バッチサイズが整数であることを確認
        print(f"バッチサイズ (型変換後): {batch_size}", flush=True)
        print(f"トレインチャンクサイズ: {len(train_chunk)}, バルチャンクサイズ: {len(val_chunk)}", flush=True)

        train_loader = DataLoader(train_chunk, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
        val_loader = DataLoader(val_chunk, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

        # チャンクの最初にモデルを保存 (weight.ptが存在しない場合のみ)
        if chunk_idx == 0 and not os.path.exists(pt_file_path):
            save_model(model, 'weight.pt')

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

                print(f"チャンク {chunk_idx + 1}/{len(train_chunks)}, エポック {epoch + 1}/{num_epochs}, バッチ {batch_idx + 1}/{len(train_loader)}, 損失: {loss.item()}")
                sys.stdout.flush()
                
            print(f"チャンク {chunk_idx + 1}/{len(train_chunks)}, エポック {epoch + 1}/{num_epochs}, 平均トレイン損失: {train_loss / len(train_loader)}")
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
            print(f"チャンク {chunk_idx + 1}/{len(train_chunks)}, エポック {epoch + 1}/{num_epochs}, 平均バル損失: {avg_val_loss}")
            sys.stdout.flush()

            scheduler.step()

            # エポック1は必ず保存
            if epoch == 0:
                best_val_loss = avg_val_loss
                save_model(model, 'weight.pt')
                print(f"エポック {epoch + 1} で初回モデルを保存, バル損失: {avg_val_loss}", flush=True)
                sys.stdout.flush()

            # 改善された場合にのみモデルを保存
            elif avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                save_model(model, 'weight.pt')
                print(f"エポック {epoch + 1} でモデルを保存, バル損失: {avg_val_loss}", flush=True)
                sys.stdout.flush()
                
            else:
                patience_counter += 1
                print(f"改善なし. エポック: {epoch + 1}, 忍耐カウンタ: {patience_counter}", flush=True)
                sys.stdout.flush()

            if chunk_idx == len(train_chunks) - 1 and patience_counter >= patience:
                print(f"最後のチャンクのエポック {epoch + 1} で早期停止します。", flush=True)
                sys.stdout.flush()
                break


def start(home_path, batch_size=4):
    image_dir = os.path.join(home_path, "image")
    mask_dir = os.path.join(home_path, "mask")

    training(image_dir, mask_dir, batch_size)

    shutil.rmtree(home_path)
