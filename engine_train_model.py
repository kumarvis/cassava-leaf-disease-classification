import tensorflow as tf
from src_train_model.img_classification_csv_input import ImageClassificationCSVDataPipeline
from config.img_classification_config import ConfigObj
from models.create_model import get_custom_model
import math

## TF dataset
data_pipeline_obj = ImageClassificationCSVDataPipeline()
dataset = data_pipeline_obj.get_tf_dataset()
train_ds, valid_ds = data_pipeline_obj.split_dataset(dataset, ConfigObj.Validation_Fraction)

## Get model
custom_model = get_custom_model()

## Prepare for training
DATASET_SIZE = data_pipeline_obj.get_num_samples()
train_steps_per_epoch = math.ceil((DATASET_SIZE * (1 - ConfigObj.Validation_Fraction) / ConfigObj.batch_size))
val_steps_per_epoch = math.ceil((DATASET_SIZE * ConfigObj.Validation_Fraction) / ConfigObj.batch_size)
num_epochs = ConfigObj.epochs

## callbacks
from callbacks.custom_callbacks import MyCallBacks

my_call_backs_obj = MyCallBacks(train_steps_per_epoch)
lst_my_callbacks = my_call_backs_obj.get_list_callbacks()

## Train Start:
history_freeze = custom_model.fit(
    train_ds, epochs=num_epochs, steps_per_epoch=train_steps_per_epoch,
    validation_data=valid_ds, validation_steps=val_steps_per_epoch,
    verbose=1, callbacks=lst_my_callbacks)

## Plottings
from src_train_model.plot_keras_hist import dump_hist_data, plot_hist_frm_csv
hist_csv_path = dump_hist_data(history_freeze, prefix='base')
plot_hist_frm_csv(hist_csv_path, prefix='base')

print('----> EXPERIMENT FINISHED <----')


