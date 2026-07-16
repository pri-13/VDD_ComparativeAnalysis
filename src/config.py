#centralized configuration file for the Vehicle Damage Classification project.
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#data directories
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

#processed classification dataset
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

#output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"
TRAINING_OUTPUT_DIR = OUTPUTS_DIR / "training"
EVALUATION_OUTPUT_DIR = OUTPUTS_DIR / "evaluation"
COMPARISON_OUTPUT_DIR = OUTPUTS_DIR / "comparison"

#saved models
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

#logs
LOGS_DIR = PROJECT_ROOT / "logs"
TRAINING_LOGS_DIR = LOGS_DIR / "training_logs"
NOTEBOOK_LOGS_DIR = LOGS_DIR / "notebook_logs"

#dataset configuration
with open(DATA_DIR / "data.yaml") as file:
    names = yaml.safe_load(file)["names"]
CLASS_NAMES = list(names.values()) if isinstance(names, dict) else names
NUM_CLASSES = len(CLASS_NAMES)

#image configuration
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3

#training configuration
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

DATASET_SPLITS = {
    "train": TRAIN_DIR,
    "val": VAL_DIR,
    "test": TEST_DIR
}

#model configuration
MODELS = ["custom_cnn", "vgg16", "mobilenetv2", "densenet121", "efficientnetb0"]