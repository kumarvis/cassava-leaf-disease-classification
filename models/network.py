import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from config.img_classification_config import ConfigObj

def create_network():
    input_shape = (ConfigObj.img_dim, ConfigObj.img_dim, ConfigObj.img_channels)
    if ConfigObj.Network_Architecture == 'ResNet50':
        feature_extractor = tf.keras.applications.resnet50.ResNet50(
        include_top=False, weights='imagenet', input_shape=input_shape)

    elif ConfigObj.Network_Architecture == 'EfficientnetB3':
        feature_extractor = tf.keras.applications.efficientnet.EfficientNetB3(
            include_top=False, weights='imagenet', input_shape=input_shape)

    num_classes = ConfigObj.Num_Classes
    features = feature_extractor.output
    avg_pool = GlobalAveragePooling2D(data_format='channels_last')(features)
    predictions = Dense(num_classes, activation='softmax')(avg_pool)

    my_newtork = Model(feature_extractor.input, predictions)
    return my_newtork