from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from src.config import *

def create_dirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

#create project directories
def create_project_dirs():
    create_dirs(
        OUTPUTS_DIR,
        EDA_OUTPUT_DIR,
        TRAINING_OUTPUT_DIR,
        EVALUATION_OUTPUT_DIR,
        COMPARISON_OUTPUT_DIR,
        SAVED_MODELS_DIR,
        LOGS_DIR,
        TRAINING_LOGS_DIR,
        NOTEBOOK_LOGS_DIR
    )