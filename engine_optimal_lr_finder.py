import os
import math
import tensorflow as tf

from src_train_model.img_classification_csv_input import ImageClassificationCSVDataPipeline
from config.img_classification_config import ConfigObj
from models.create_model import get_custom_model
from src_optimal_lr_finder.LRFinder import LRFinder

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

## LR Finder
min_lr, max_lr = 0.01, 0.5
lr_finder = LRFinder(min_lr, max_lr)
lr_log_file_path = os.path.join(ConfigObj.Path_Parent_Dir, 'src_optimal_lr_finder', 'optimal_lr.log')

history_freeze = custom_model.fit(
    train_ds, epochs=num_epochs, steps_per_epoch=train_steps_per_epoch,
    validation_data=valid_ds, validation_steps=val_steps_per_epoch,
    verbose=False, callbacks=[lr_finder])

lr_finder.dump(lr_log_file_path)
print('----> LR EXPERIMENT FINISHED <----')