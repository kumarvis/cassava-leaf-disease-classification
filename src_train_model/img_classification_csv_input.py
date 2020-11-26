from __future__ import print_function
import os
import csv
from PIL import Image
import tensorflow as tf
import cv2
from pathlib import Path
from config.img_classification_config import ConfigObj

class ImageClassificationCSVDataPipeline:
    def __init__(self):
        self.num_samples = 0
        self.number_parallel_calls = 1 #tf.data.experimental.AUTOTUNE
        self.batch_size = (ConfigObj.batch_size_per_gpu * ConfigObj.Number_GPU)

    def get_num_samples(self):
        if self.num_samples == 0:
            ## Assumption all the images are in one ground truth file
            assert os.path.exists(ConfigObj.Gt_Path), (
                "Cannot find Ground truth csv file {}.".format(ConfigObj.Gt_Path))

            with open(ConfigObj.Gt_Path) as gt_csv:
                csv_reader = csv.reader(gt_csv, delimiter=',')
                self.num_samples = len(list(csv_reader)) - 1 ## -1 for header
            return self.num_samples
        else:
            return self.num_samples

    def get_samples_fn(self):
        ## Assumption all the images are in one ground truth file
        assert os.path.exists(ConfigObj.Gt_Path), (
            "Cannot find Ground truth csv file {}.".format(ConfigObj.Gt_Path))

        images_path = []
        labels = []
        with open(ConfigObj.Gt_Path) as gt_csv:
            csv_reader = csv.reader(gt_csv, delimiter=',')
            next(csv_reader)  ##skip headers
            for sample in csv_reader:
                if Path(sample[0]).suffix == ConfigObj.Img_Ext:
                    img_name = sample[0]
                else:
                    img_name = sample[0] + ConfigObj.Img_Ext

                img_path = os.path.join(ConfigObj.Path_Parent_Dir, 'input',
                                        'cassava-leaf-disease-classification', 'train_images', img_name)
                img_lbl = int(sample[1])
                images_path.append(img_path)
                labels.append(img_lbl)

        return (images_path, labels)

    def cv2_func(self, img_path):
        img_path = img_path.numpy().decode("utf-8")
        img = cv2.imread(img_path)
        #img = cv2.resize(img, (ConfigObj.img_size, ConfigObj.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_tensor = tf.cast(img/255.0, dtype=tf.float32)
        aug_img = tf.image.resize(rgb_tensor, size=[ConfigObj.img_dim, ConfigObj.img_dim])
        return aug_img

    def tf_cv2_parsefn(self, image_path, label):
        """Parse a single input sample
        """
        [aug_img] = tf.py_function(self.cv2_func, [image_path], [tf.float32])
        label = tf.cast(label, dtype=tf.float32)
        return (aug_img, label)

    def set_shapes(self, img, label, img_shape):
        img.set_shape(img_shape)
        return img, label

    def get_tf_dataset(self):
        samples = self.get_samples_fn()
        dataset = tf.data.Dataset.from_tensor_slices(samples)
        img_shape = (ConfigObj.img_dim, ConfigObj.img_dim, ConfigObj.img_channels)

        if ConfigObj.Mode == 'train':
            dataset = dataset.shuffle(self.get_num_samples())
            dataset = dataset.repeat()
            dataset = dataset.map(lambda img_path, label: self.tf_cv2_parsefn(img_path, label),
                                  self.number_parallel_calls)
            ###Using tf.py_function lost the shape info below is the hack for it
            '''
            Ref URL:
            https://github.com/tensorflow/tensorflow/issues/31373
            
            '''
            dataset = dataset.map(lambda img, label: self.set_shapes(img, label, img_shape))
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            return dataset

    def split_dataset(self, dataset: tf.data.Dataset, validation_data_fraction: float):
        """
        Splits a dataset of type tf.data.Dataset into a training and validation dataset using given ratio. Fractions are
        rounded up to two decimal places.
        @param dataset: the input dataset to split.
        @param validation_data_fraction: the fraction of the validation data as a float between 0 and 1.
        @return: a tuple of two tf.data.Datasets as (training, validation)
        Refrence URL:
        https://stackoverflow.com/questions/59669413/
        what-is-the-canonical-way-to-split-tf-dataset-into-test-and-validation-subsets
        """

        validation_data_percent = round(validation_data_fraction * 100)
        if not (0 <= validation_data_percent <= 100):
            raise ValueError("validation data fraction must be ∈ [0,1]")

        dataset = dataset.enumerate()
        train_dataset = dataset.filter(lambda f, data: f % 100 > validation_data_percent)
        validation_dataset = dataset.filter(lambda f, data: f % 100 <= validation_data_percent)

        # remove enumeration
        train_dataset = train_dataset.map(lambda f, data: data)
        validation_dataset = validation_dataset.map(lambda f, data: data)

        return train_dataset, validation_dataset



