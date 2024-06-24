import torch
import sys
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn
import os

from model import FastFPN, YOLOv8Backbone, resize_size

# 推論用のデータ変換
image_transform = transforms.Compose([
    transforms.Resize((resize_size, resize_size)),
    transforms.ToTensor()
])


def DumpFile(in_path):
    image_files = []
    if os.path.isfile(in_path):
        if in_path.lower().endswith(('.jpg', '.png')):
            image_files.append(in_path)
    else:
        for root, dirs, files in os.walk(in_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    image_files.append(os.path.join(root, file))
    return image_files


# バックボーンを設定
def load_model(model_path):
    model = FastFPN()
    backbone = YOLOv8Backbone('yolov8n.pt')
    model.backbone = backbone

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    model.float()
    return model


def infer(model, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 画像の読み込みと変換
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    # 推論の実行
    with torch.no_grad():
        detections, masks = model(image_tensor)

    # マスクの後処理
    masks = torch.sigmoid(masks)
    masks = masks.squeeze(0).cpu().numpy()

    # マスクを元の画像サイズにリサイズ
    resized_masks = []
    for mask in masks:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize(original_size, Image.BILINEAR)
        resized_masks.append(np.array(mask_img) / 255.0)
    resized_masks = np.array(resized_masks)

    return resized_masks


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


def apply_mask(image, mask, alpha=0.5, color=(0, 1, 0), threshold=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask > threshold, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def plot_results(image_path, masks, class_names=None, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    num_classes = masks.shape[0]
    for i in range(num_classes):
        masked_image = apply_mask(image.copy(), masks[i], threshold=threshold)


def make_background_transparent(image_path, dir_path, mask, threshold=0.5):
    image = Image.open(image_path).convert("RGBA")
    mask = (mask > threshold).astype(np.uint8) * 255

    image_data = np.array(image)
    alpha_channel = mask[:, :, None]

    # 元の画像のアルファチャネルをマスクに置き換える
    image_data[:, :, 3] = alpha_channel[:, :, 0]

    output_image = Image.fromarray(image_data)

    # ファイル名を生成
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    new_name = f"{name}.png"
    out_path = os.path.join(dir_path, new_name)
    output_image.save(out_path)

    print(f"Saved: {out_path}")
    sys.stdout.flush()


def main():
    if len(sys.argv) != 3:
        print("Please check the data")
        sys.stdout.flush()
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    
    # モデルのロード
    model_path = 'weight.pt'
    model = load_model(model_path)

    image_paths = DumpFile(in_path)
    filtered_paths = [path for path in image_paths if not os.path.basename(path).startswith('.')]
    filtered_paths.sort()

    for image_path in filtered_paths:
        # 推論の実行
        masks = infer(model, image_path)

        # 結果の表示
        plot_results(image_path, masks, threshold=0.3)

        # 背景を透明にする
        make_background_transparent(image_path, out_path, masks[0], threshold=0.3)

    

if __name__ == "__main__":
    main()
