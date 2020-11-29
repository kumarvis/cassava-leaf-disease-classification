import os
import time
import tensorflow as tf
from config.img_classification_config import ConfigObj
from callbacks.lr_tracker import LRTrackerCallback
from callbacks.one_cycle_policy import OneCycleScheduler

class MyCallBacks:
    def __init__(self, steps):
        self.lr_max = ConfigObj.learning_rate
        self.steps = steps
    def get_list_callbacks(self):
        plot_path = os.path.join(ConfigObj.Path_Parent_Dir, 'src_train_model', 'plots_logs')
        csv_logger_name = time.strftime("%Y%m%d-%H%M%S") + '.csv'
        csv_logger_path = os.path.join(plot_path, csv_logger_name)

        ## framework callbacks
        csv_logger_callback = tf.keras.callbacks.CSVLogger(csv_logger_path)

        tensor_board_callback = tf.keras.callbacks.TensorBoard(log_dir=plot_path, histogram_freq=0, write_graph=True)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(ConfigObj.CheckPoints_Dir, 'weights-epoch{epoch:02d}-loss{val_loss:.2f}.h5'),
            monitor='val_accuracy', verbose=1, period=ConfigObj.model_chkpoint_period, mode='auto')

        tf.keras.callbacks.EarlyStopping(patience=ConfigObj.early_stopping_patience, monitor='val_acc',
                                         restore_best_weights=True)

        my_callbacks = [csv_logger_callback, tensor_board_callback, model_checkpoint_callback]
        ## custom callbacks
        if ConfigObj.one_cycle_lr_policy:
            one_cycle_policy_callback = OneCycleScheduler(self.lr_max, self.steps)
            my_callbacks.append(one_cycle_policy_callback)

        return my_callbacks


