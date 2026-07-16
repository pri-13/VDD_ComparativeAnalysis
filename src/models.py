import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from src.config import *

#create classifier head
def classifier_head(base_model, preprocess_fn):
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, IMAGE_CHANNELS))
    x = preprocess_fn(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)
    return tf.keras.Model(inputs, outputs)

#create custom cnn
def build_custom_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMAGE_SIZE, IMAGE_CHANNELS)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

#create vgg16 model
def build_vgg16():
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS)
    )
    return classifier_head(base_model, vgg16_preprocess)


#create mobilenetv2 model
def build_mobilenetv2():
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS)
    )
    return classifier_head(base_model, mobilenet_preprocess)


#create densenet121 model
def build_densenet121():
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS)
    )
    return classifier_head(base_model, densenet_preprocess)


#create efficientnetb0 model
def build_efficientnetb0():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS)
    )
    return classifier_head(base_model, efficientnet_preprocess)

#compile keras model
def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

#build selected model
def get_model(model_name):
    models = {
        "custom_cnn": build_custom_cnn,
        "vgg16": build_vgg16,
        "mobilenetv2": build_mobilenetv2,
        "densenet121": build_densenet121,
        "efficientnetb0": build_efficientnetb0
    }
    return compile_model(models[model_name]())