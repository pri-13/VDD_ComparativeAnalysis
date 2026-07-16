from pathlib import Path
import shutil

#create class folders
def create_class_folders(output_dir, class_names):
    for split in ["train", "val", "test"]:
        for class_name in class_names:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

#get class ids from label file
def get_class_ids(label_path):
    with open(label_path) as file:
        return sorted({int(line.split()[0]) for line in file if line.strip()})

#copy image into corresponding class folders
def copy_image(image_path, output_dir, split, class_names, class_ids):
    for class_id in class_ids:
        shutil.copy2(image_path, output_dir / split / class_names[class_id] / image_path.name)

#convert yolo dataset to classification dataset
def convert_dataset(data_dir, output_dir, class_names):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    create_class_folders(output_dir, class_names)
    for split in ["train", "val", "test"]:
        image_dir = data_dir / split / "images"
        label_dir = data_dir / split / "labels"
        for image_path in image_dir.glob("*.*"):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            class_ids = get_class_ids(label_path)
            copy_image(image_path, output_dir, split, class_names, class_ids)