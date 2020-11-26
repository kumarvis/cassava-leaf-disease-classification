import tensorflow as tf
from .tf_rectified_adam import RectifiedAdam
from config.img_classification_config import ConfigObj

def create_optimizer(learning_rate):
    # Setup optimizer
    if ConfigObj.optimizer == "adadelta":
      optimizer = tf.keras.optimizers.Adadelta(
          learning_rate=learning_rate)
    elif ConfigObj.optimizer == "adagrad":
      optimizer = tf.keras.optimizers.Adagrad(
          learning_rate=learning_rate)
    elif ConfigObj.optimizer == "adam":
      optimizer = tf.keras.optimizers.Adam(
          learning_rate=learning_rate)
    elif ConfigObj.optimizer == "radam":
      optimizer = RectifiedAdam(
          learning_rate=learning_rate)
    elif ConfigObj.optimizer == "ftrl":
      optimizer = tf.keras.optimizers.Ftrl(
          learning_rate=learning_rate)
    elif ConfigObj.optimizer == "sgd":
      optimizer = tf.keras.optimizers.SGD(
          learning_rate=learning_rate,
          momentum=0.9,
          name="Momentum")
    elif ConfigObj.optimizer == "rmsprop":
      optimizer = tf.keras.optimizers.RMSProp(
          learning_rate=learning_rate)
    elif ConfigObj.optimizer == "sgd":
      optimizer = tf.keras.optimizers.GradientDescent(
        learning_rate=learning_rate)
    else:
      raise ValueError("Optimizer [%s] was not recognized" %
                       ConfigObj.optimizer)
    return optimizer