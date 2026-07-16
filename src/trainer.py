import gc
import time
import pandas as pd
import tensorflow as tf
from src.config import *

#create callbacks
def get_callbacks(model_name):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=SAVED_MODELS_DIR / f"{model_name}.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            TRAINING_LOGS_DIR / f"{model_name}.log"
        )
    ]

#train keras model
def train_model(
    model,
    train_dataset,
    validation_dataset,
    model_name
):
    start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=get_callbacks(model_name),
        verbose=1
    )
    training_time = round(time.time() - start_time, 2)
    info = pd.DataFrame({
        "Model": [model_name],
        "Epochs": [len(history.history["loss"])],
        "Best Train Accuracy": [max(history.history["accuracy"])],
        "Best Validation Accuracy": [max(history.history["val_accuracy"])],
        "Best Train Loss": [min(history.history["loss"])],
        "Best Validation Loss": [min(history.history["val_loss"])],
        "Training Time (sec)": [training_time]
    })
    info.to_csv(
        TRAINING_OUTPUT_DIR / f"{model_name}_training_info.csv",
        index=False
    )
    return history

#save training history
def save_history(history, model_name):
    pd.DataFrame(history.history).to_csv(
        TRAINING_OUTPUT_DIR / f"{model_name}_history.csv",
        index=False
    )

#free memory
def clear_memory(model=None):
    if model is not None:
        del model
    tf.keras.backend.clear_session()
    gc.collect()