import os
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
from config.img_classification_config import ConfigObj

class LRTrackerCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.lst_lr_per_batch = []
        self.lst_lr_per_epoch = []

    def on_batch_end(self, batch, logs=None):
        lr = self.model.optimizer.lr.read_value()
        self.lst_lr_per_batch.append(lr)

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr.read_value()
        self.lst_lr_per_epoch.append(lr)

    def on_train_end(self, logs=None):
        plot_path = os.path.join(ConfigObj.Path_Parent_Dir, 'src_train_model', 'plots_logs')
        lr_labels = ['batch-lr-rate', 'epoch-lr-rate']
        lr_csv_path = os.path.join(plot_path,  "lr_rate_log.csv")
        ## Backup to csv file
        with open(lr_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(lr_labels)
            writer.writerow(self.lst_lr_per_batch)
            writer.writerow(self.lst_lr_per_epoch)

        ## Plot lr-rate vs batch
        plt.plot(self.lst_lr_per_batch)
        plt.title('batch-lr-rate')
        plt.ylabel('lr-rate')
        plt.xlabel('batch')
        batch_lr_fig_path = os.path.join(plot_path, 'batch-lr.png')
        plt.savefig(batch_lr_fig_path)
        plt.clf()  ##clear the plot

        ## Plot lr-rate vs epochs
        plt.plot(self.lst_lr_per_epoch)
        plt.title('epoch-lr-rate')
        plt.ylabel('lr-rate')
        plt.xlabel('epoch')
        epoch_lr_fig_path = os.path.join(plot_path, 'epoch-lr.png')
        plt.savefig(epoch_lr_fig_path)



