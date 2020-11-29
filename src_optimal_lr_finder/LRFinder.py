import os
import sys
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import logging
import matplotlib.pyplot as plt
from pathlib import Path

from config.img_classification_config import ConfigObj

class LRFinder(Callback):
    def __init__(self, start_lr, end_lr, max_steps: int = 100, smoothing=0.9):
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_begin(self, logs=None):
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []
    
    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.exp_annealing(self.step)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)
    
    def on_train_batch_end(self, batch, logs=None):
        debug = True
        logs = logs or {}
        loss = logs.get('loss')
        step = self.step
        if loss:
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)
            self.lrs.append(self.lr)

            if step == 0 or loss < self.best_loss:
                self.best_loss = loss

            if smooth_loss > 4 * self.best_loss or tf.math.is_nan(smooth_loss):
                self.model.stop_training = True

        if step == self.max_steps:
            self.model.stop_training = True

        if debug:
                print("[DEBUG] Step {:d}, Smooth Loss {:.8f}, LR {:.8f}".format(
                    step, smooth_loss, self.model.optimizer.lr.numpy()))

        self.step += 1

    def exp_annealing(self, step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)

    def dump(self, log_file_path):
        if os.path.exists(log_file_path) == True:
            os.remove(log_file_path)
        logging.basicConfig(filename=log_file_path, filemode='w', format='%(message)s',
                            level=logging.INFO)

        logging.info('Loss_Each_Iteration')
        loss_str = ",".join([str(item) for item in self.losses])
        logging.info(loss_str)

        logging.info('Learning_Rate_Each_Iteration')
        lr_str = ",".join([str(item) for item in self.lrs])
        logging.info(lr_str)

    def plot_smooth_curve(self, log_file_name):
        file = open(log_file_name, 'r')
        Lines = file.readlines()
        loss_string = Lines[1].strip().split(',')
        loss_values = [float(vv) for vv in loss_string]
        lst_rate_loss = [(loss_values[i] - loss_values[i+1]) for i in range(len(loss_values) - 1)]
        lst_rate_loss.insert(0, 0)
        loss_grad = lst_rate_loss # np.gradient(loss_values)
        lst_lr_string = Lines[3].strip().split(',')
        lst_lr_values = [float(vv) for vv in lst_lr_string]
        ## skip first k values and the last k,
        ## focus on the interesting parts of the graph
        #plt.xticks(lst_lr_values[5:-5])
        index1, index2 = 5, 10
        plt.plot(lst_lr_values[index1:-index2], loss_values[index1:-index2], 'r')
        plt.plot(lst_lr_values[index1:-index2], loss_grad[index1:-index2], 'g')

        plt.xlabel("learning-rate")
        plt.ylabel("loss")
        plt.xscale("log")
        plt.legend(['model-loss', 'grad-loss'], loc='lower right')
        plt.grid()
        plot_path = os.path.join(str(Path(__file__).parent.absolute()), 'loss_lr_rate.png')
        print(plot_path)
        plt.savefig(plot_path)
        print('plot_smooth_curve exit')

min_lr, max_lr = 0.01, 0.1
lr_finder = LRFinder(min_lr, max_lr)
lr_log_file_path = os.path.join(ConfigObj.Path_Parent_Dir, 'src_optimal_lr_finder', 'optimal_lr.log')
lr_finder.plot_smooth_curve(lr_log_file_path)

