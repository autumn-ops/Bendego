import os
import sys
import subprocess
import glob
import random
from PIL import Image, ImageFilter, ImageDraw, ImageChops
import numpy as np
import cv2

from train import start

def resize_image(image, target_size):
    width, height = image.size
    if width > height:
        new_width = target_size
        new_height = int((target_size / width) * height)
    else:
        new_height = target_size
        new_width = int((target_size / height) * width)
    return image.resize((new_width, new_height), Image.LANCZOS)

def create_shadow(original_image):
    # ランダムなシフト量とアルファ値を生成
    shift_x = random.randint(5, 15)
    shift_y = random.randint(5, 15)
    alpha_value = random.randint(80, 150)
    blur_radius = random.randint(5, 15)

    shifted_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
    shifted_image.paste(original_image, (shift_x, shift_y))

    data = shifted_image.getdata()
    new_data = []
    for item in data:
        if item[3] > 0:
            new_data.append((0, 0, 0, alpha_value))
        else:
            new_data.append(item)
    shifted_image.putdata(new_data)

    blurred_shadow = shifted_image.filter(ImageFilter.GaussianBlur(blur_radius))
    return blurred_shadow

def create_background(root_back_paths, size):
    root_back_path = random.choice(root_back_paths)
    root_image = Image.open(root_back_path).convert("RGBA")
    root_width, root_height = root_image.size
    crop_width, crop_height = size

    if crop_width > root_width or crop_height > root_height:
        raise ValueError("指定されたサイズが元の背景画像より大きいです。")
        sys.stdout.flush()

    x = random.randint(0, root_width - crop_width)
    y = random.randint(0, root_height - crop_height)

    background = root_image.crop((x, y, x + crop_width, y + crop_height))
    background = add_gray_noise(background)
    return background

def mask(original_image):
    data = np.array(original_image)
    alpha_channel = data[:, :, 3]

    # 元の画像と同じサイズの黒い背景を作成
    black_bg = np.zeros_like(data)
    black_bg[:, :, 3] = alpha_channel

    # アルファチャネルが0以外のマスクイメージを作成
    mask_data = np.where(alpha_channel[:, :, None] > 0, [255, 255, 255, 255], [0, 0, 0, 255])

    gray_image = cv2.cvtColor(mask_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    num_labels, labels_im = cv2.connectedComponents(thresh)

    color_map = []
    for i in range(1, num_labels):
        color = [random.randint(0, 255) for _ in range(3)] + [255]
        color_map.append((i, color))

    new_data = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
    for label, color in color_map:
        new_data[labels_im == label] = color

    return Image.fromarray(new_data, "RGBA")

def add_gray_noise(image, mean=0, std=4):
    alpha = image.split()[3]
    gray_image = image.convert('L')
    noise = np.random.normal(mean, std, (gray_image.height, gray_image.width))
    np_gray = np.array(gray_image)
    np_gray = np.clip(np_gray + noise, 0, 255)
    noisy_image = Image.fromarray(np_gray.astype('uint8')).convert("RGBA")
    noisy_image.putalpha(alpha)
    return noisy_image

def random_shift(image, background_size, max_shift=10):
    bg_width, bg_height = background_size
    img_width, img_height = image.size
    max_x_shift = bg_width - img_width
    max_y_shift = bg_height - img_height
    x_shift = random.randint(0, max_x_shift)
    y_shift = random.randint(0, max_y_shift)
    shifted_image = Image.new("RGBA", (bg_width, bg_height), (0, 0, 0, 0))
    shifted_image.paste(image, (x_shift, y_shift))
    return shifted_image, x_shift, y_shift

def main():
    input_folder = sys.argv[1]
    out_path = sys.argv[2]
    batch_size = sys.argv[3]
    
    home_path = os.path.join(os.path.expanduser("~"), "Dataset")
    
    background_folder = "screen"

    os.makedirs(home_path, exist_ok=True)

    image_paths = glob.glob(os.path.join(input_folder, '**', '*.png'), recursive=True)
    filtered_paths = [path for path in image_paths if not os.path.basename(path).startswith('.')]
    filtered_paths.sort()
    print(len(filtered_paths))
    sys.stdout.flush()

    back_paths = glob.glob(os.path.join(background_folder, '*.jpg'))
    root_back_paths = [path for path in back_paths if not os.path.basename(path).startswith('.')]

    for index, img in enumerate(filtered_paths):
        print(f"{index + 1}/{len(filtered_paths)}: {img}")
        sys.stdout.flush()
        filename, _ = os.path.splitext(os.path.basename(img))
        out_path_img = os.path.join(home_path, "image\\" + filename + "_image.jpg")
        out_path_mask = os.path.join(home_path, "mask\\" + filename + "_mask.jpg")

        original_image = Image.open(img).convert("RGBA")
        resized_image = resize_image(original_image, 1024)
        shadow_image = create_shadow(resized_image)
        background = create_background(root_back_paths, resized_image.size)

        # オブジェクトごとにランダムな色のマスクを生成
        mask_image = mask(resized_image)

        shifted_image, x_shift, y_shift = random_shift(resized_image, resized_image.size)

        # 背景画像を合成
        new_image = background.copy()
        new_image.paste(shadow_image, (0, 0), shadow_image)
        new_image.paste(shifted_image, (x_shift, y_shift), shifted_image)

        new_image = new_image.convert('RGB')
        mask_image = mask_image.convert('RGB')

        os.makedirs(os.path.dirname(out_path_img), exist_ok=True)
        os.makedirs(os.path.dirname(out_path_mask), exist_ok=True)

        new_image.save(out_path_img, 'JPEG', quality=100)
        mask_image.save(out_path_mask, 'JPEG', quality=100)
        
    start(home_path, batch_size)

if __name__ == "__main__":
    main()
