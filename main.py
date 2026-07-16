from src.config import DATA_DIR, PROCESSED_DATA_DIR, CLASS_NAMES
from src.preprocessing import convert_dataset

if __name__ == "__main__":
    convert_dataset(DATA_DIR, PROCESSED_DATA_DIR, CLASS_NAMES)
    print("Dataset converted successfully.")