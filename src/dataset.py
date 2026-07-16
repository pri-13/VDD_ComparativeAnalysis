from pathlib import Path
from PIL import Image
import yaml
import tensorflow as tf
from src.config import *


#load class names from data.yaml
def load_classes(yaml_path):
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)["names"]

#count total images in a directory
def count_images(image_dir):
    return len(list(Path(image_dir).glob("*.*")))

#check corrupted images
def verify_images(image_dir):
    corrupted = []
    for image_path in Path(image_dir).glob("*.*"):
        try:
            Image.open(image_path).verify()
        except Exception:
            corrupted.append(image_path)
    return corrupted

#create tensorflow dataset
def create_dataset(directory, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=RANDOM_SEED
    )


#optimize dataset pipeline
def optimize_dataset(dataset):
    return dataset.prefetch(tf.data.AUTOTUNE)