import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import math
import logging
import math
from pathlib import Path

class LRFinder(object):
    def __init__(self, model, optimizer, loss_fn, dataset):
        super(LRFinder, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dataset = dataset
        # placeholders
        #self.lrs = None
        self.loss_values = None
        self.lr_values = None
        self.min_lr = None
        self.max_lr = None
        self.num_itrs = None
        self.smoothing_factor = 0.05
        self.beta = 0.98

    @tf.function
    def train_step(self, x, y, curr_lr):
        tf.keras.backend.set_value(self.optimizer.lr, curr_lr)
        with tf.GradientTape() as tape:
            # forward pass
            y_ = self.model(x)
            # external loss value for this batch
            loss = self.loss_fn(y, y_)
            # add any losses created during forward pass
            loss += sum(self.model.losses)
            # get gradients of weights wrt loss
            grads = tape.gradient(loss, self.model.trainable_weights)
        # update weights
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    def range_test(self, min_lr, max_lr, num_itrs=100, debug=True):
        # create learning rate schedule
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_itrs = num_itrs
        #self.lrs = np.linspace(self.min_lr, self.max_lr, num=self.num_itrs)
        self.loss_values, self.lr_values = [], []
        curr_lr = self.min_lr

        avg_loss = 0
        best_loss = math.inf
        coff = (max_lr / min_lr) ** (1/num_itrs)
        # initialize loss_values, lr_values
        for step, (x, y) in enumerate(self.dataset):
            step_plus = step + 1
            if step_plus >= self.num_itrs:
                break
            loss = self.train_step(x, y, curr_lr)
            curr_loss = loss.numpy()
            # Compute the smoothed loss
            avg_loss = self.beta * avg_loss + (1 - self.beta) * curr_loss
            smoothed_loss = avg_loss / (1 - self.beta ** step_plus)
            # Stop if the loss is exploding after certain number of itrs
            if step_plus > 10 and smoothed_loss > 4 * best_loss:
                print('Loss Expload Exit LR finder')
                break
            # Record the best loss
            if smoothed_loss < best_loss or step_plus == 1:
                best_loss = smoothed_loss

            self.loss_values.append(smoothed_loss)
            self.lr_values.append(curr_lr)

            if debug:
                print("[DEBUG] Step {:d}, Smooth Loss {:.8f}, LR {:.8f}".format(
                    step, smoothed_loss, self.optimizer.learning_rate.numpy()))

            curr_lr = curr_lr * coff

    def dump(self, log_file_path):
        if os.path.exists(log_file_path) == True:
            os.remove(log_file_path)
        logging.basicConfig(filename=log_file_path, filemode='w', format='%(message)s',
                            level=logging.INFO)

        logging.info('Loss_Each_Iteration')
        loss_str = ",".join([str(item) for item in self.loss_values])
        logging.info(loss_str)

        logging.info('Learning_Rate_Each_Iteration')
        lr_str = ",".join([str(item) for item in self.lr_values])
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

