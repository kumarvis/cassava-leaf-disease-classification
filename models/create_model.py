import tensorflow as tf
from .create_optimizer import create_optimizer
from .network import create_network
from config.img_classification_config import ConfigObj

def get_custom_model():
    lr = ConfigObj.learning_rate

    optimizer = create_optimizer(lr)
    custom_network_model = create_network()
    # Compile:
    custom_network_model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc')
        ])

    return custom_network_model
