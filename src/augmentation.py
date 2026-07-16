import tensorflow as tf

#create augmentation pipeline
def get_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])

#create normalization layer
def get_normalization():
    return tf.keras.layers.Rescaling(1.0 / 255)