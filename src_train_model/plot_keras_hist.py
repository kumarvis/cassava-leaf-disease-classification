import os
from matplotlib import pyplot as plt
import pathlib
import csv
from pathlib import Path
from config.img_classification_config import ConfigObj
import pandas as pd

def dump_hist_data(history, prefix):
    plot_path = os.path.join(ConfigObj.Path_Parent_Dir, 'src_train_model', 'plots_logs')
    hist_lbls = ['train_acc', 'val_acc', 'loss', 'val_loss']
    train_accuracy = history.history['acc']
    valid_accuracy = history.history['val_acc']
    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']

    hist_dict = {'train_acc': train_accuracy, 'val_acc': valid_accuracy,
                 'train_loss': train_loss, 'val_loss': valid_loss}
    hist_df = pd.DataFrame(hist_dict)
    hist_csv_path = os.path.join(plot_path, prefix + "_" + "hist_log.csv")
    hist_df.to_csv(hist_csv_path, index=False)

    return hist_csv_path

def plot_hist_frm_csv(hist_csv_path, prefix):
    plot_path = os.path.join(ConfigObj.Path_Parent_Dir, 'src_train_model', 'plots_logs')
    hist_df = pd.read_csv(hist_csv_path)
    ##accuracies
    lst_train_acc = hist_df['train_acc'].tolist()
    lst_val_acc = hist_df['val_acc'].tolist()
    plt.plot(lst_train_acc)
    plt.plot(lst_val_acc)
    plt.title('model-accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    model_acc_fig_path = os.path.join(plot_path, prefix + '-' + 'model-accuracy.png')
    plt.savefig(model_acc_fig_path)
    plt.clf()  ##clear the plot
    ## loss
    lst_train_loss = hist_df['train_loss'].tolist()
    lst_val_loss = hist_df['val_loss'].tolist()
    plt.plot(lst_train_loss)
    plt.plot(lst_val_loss)
    plt.title('model-loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    model_loss_fig_path = os.path.join(plot_path, prefix + '-' + 'model-loss.png')
    plt.savefig(model_loss_fig_path)
    print('Exit: plot_hist_frm_csv')

##test api
##plot_hist_frm_csv('plots_logs/base_hist_log.csv', 'TRY')