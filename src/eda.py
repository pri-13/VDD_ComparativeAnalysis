import hashlib
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


#generate dataset summary
def dataset_summary(data_dir, class_names):
    summary = []
    for split in ("train", "val", "test"):
        row = {"Split": split}
        for class_name in class_names:
            row[class_name] = len(list((data_dir / split / class_name).glob("*.*")))
        row["Total"] = sum(row[c] for c in class_names)
        summary.append(row)
    return pd.DataFrame(summary)

#display sample images
def show_samples(data_dir, class_names, samples=3):
    fig, axes = plt.subplots(len(class_names), samples, figsize=(4 * samples, 3 * len(class_names)))
    if len(class_names) == 1:
        axes = np.array([axes])
    for row, class_name in enumerate(class_names):
        images = list((data_dir / "train" / class_name).glob("*.*"))
        if not images:
            continue
        selected = random.sample(images, min(samples, len(images)))
        for col, image_path in enumerate(selected):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(image)
            axes[row, col].set_title(class_name)
            axes[row, col].axis("off")
    plt.tight_layout()
    return fig

#collect image dimensions
def image_dimensions(data_dir, class_names):
    dimensions = []
    for split in ("train", "val", "test"):
        for class_name in class_names:
            for image_path in (data_dir / split / class_name).glob("*.*"):
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                height, width = image.shape[:2]
                dimensions.append([
                    split,
                    class_name,
                    width,
                    height,
                    width / height
                ])
    return pd.DataFrame(
        dimensions,
        columns=[
            "Split",
            "Class",
            "Width",
            "Height",
            "Aspect Ratio"
        ]
    )

#compute rgb histogram
def rgb_histogram(data_dir, class_names):
    red = np.zeros(256, dtype=np.int64)
    green = np.zeros(256, dtype=np.int64)
    blue = np.zeros(256, dtype=np.int64)
    for split in ("train", "val", "test"):
        for class_name in class_names:
            for image_path in (data_dir / split / class_name).glob("*.*"):
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                red += np.histogram(image[:, :, 0], bins=256, range=(0, 256))[0]
                green += np.histogram(image[:, :, 1], bins=256, range=(0, 256))[0]
                blue += np.histogram(image[:, :, 2], bins=256, range=(0, 256))[0]
    return red, green, blue

#compute grayscale histogram
def pixel_histogram(data_dir, class_names):
    histogram = np.zeros(256, dtype=np.int64)
    for split in ("train", "val", "test"):
        for class_name in class_names:
            for image_path in (data_dir / split / class_name).glob("*.*"):
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                histogram += np.histogram(image, bins=256, range=(0, 256))[0]
    return histogram

#calculate dataset mean and std
def dataset_statistics(data_dir, class_names):
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    total_pixels = 0
    for split in ("train", "val", "test"):
        for class_name in class_names:
            for image_path in (data_dir / split / class_name).glob("*.*"):
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                pixels = image.reshape(-1, 3)
                pixel_sum += pixels.sum(axis=0)
                pixel_sq_sum += (pixels ** 2).sum(axis=0)
                total_pixels += len(pixels)
    mean = pixel_sum / total_pixels
    std = np.sqrt(pixel_sq_sum / total_pixels - mean ** 2)
    return mean, std

#detect corrupted images
def corrupted_images(data_dir, class_names):
    corrupted = []
    for split in ("train", "val", "test"):
        for class_name in class_names:
            for image_path in (data_dir / split / class_name).glob("*.*"):
                try:
                    Image.open(image_path).verify()
                except Exception:
                    corrupted.append(str(image_path))
    return corrupted

#detect duplicate images
def duplicate_images(data_dir, class_names):
    hashes = {}
    duplicates = []
    for split in ("train", "val", "test"):
        for class_name in class_names:
            for image_path in (data_dir / split / class_name).glob("*.*"):
                md5 = hashlib.md5()
                with open(image_path, "rb") as file:
                    while True:
                        chunk = file.read(8192)
                        if not chunk:
                            break
                        md5.update(chunk)
                file_hash = md5.hexdigest()
                if file_hash in hashes:
                    duplicates.append((hashes[file_hash], str(image_path)))
                else:
                    hashes[file_hash] = str(image_path)
    return duplicates