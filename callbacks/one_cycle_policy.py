import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os
from config.img_classification_config import ConfigObj

class CosineAnnealer:
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0

    def step(self):
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2. * cos


class OneCycleScheduler(Callback):
    """ `Callback` that schedules the learning rate on a 1cycle policy as per Leslie Smith's paper(https://arxiv.org/pdf/1803.09820.pdf).
    If the model supports a momentum parameter, it will also be adapted by the schedule.
    The implementation adopts additional improvements as per the fastai library: https://docs.fast.ai/callbacks.one_cycle.html, where
    only two phases are used and the adaptation is done using cosine annealing.
    In phase 1 the LR increases from `lr_max / div_factor` to `lr_max` and momentum decreases from `mom_max` to `mom_min`.
    In the second phase the LR decreases from `lr_max` to `lr_max / (div_factor * 1e4)` and momemtum from `mom_max` to `mom_min`.
    By default the phases are not of equal length, with the phase 1 percentage controlled by the parameter `phase_1_pct`.

    Reference: https://github.com/kumarvis/My-TechNotes/blob/master/Notes4-MLDL/Understanding-Research-Papers/
    VisionPapers/SuperConvergence-1CyclePolicy.md
    Reference: https://www.avanwyk.com/tensorflow-2-super-convergence-with-the-1cycle-policy/

    """

    def __init__(self, lr_max, steps, mom_min=0.85, mom_max=0.95, phase_1_pct=0.3, div_factor=25.):
        super(OneCycleScheduler, self).__init__()
        lr_min = lr_max / div_factor
        final_lr = lr_max / (div_factor * 1e4)
        phase_1_steps = int(steps * phase_1_pct)
        phase_2_steps = steps - phase_1_steps

        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0

        self.phases = [[CosineAnnealer(lr_min, lr_max, phase_1_steps),
                        CosineAnnealer(mom_max, mom_min, phase_1_steps)],
                       [CosineAnnealer(lr_max, final_lr, phase_2_steps),
                        CosineAnnealer(mom_min, mom_max, phase_2_steps)]]

        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0

        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)

    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())
        self.moms.append(self.get_momentum())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1

        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().step())

    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def get_momentum(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.momentum)
        except AttributeError:
            return None

    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass  # ignore

    def set_momentum(self, mom):
        try:
            tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
        except AttributeError:
            pass  # ignore

    def lr_schedule(self):
        return self.phases[self.phase][0]

    def mom_schedule(self):
        return self.phases[self.phase][1]

    def on_train_end(self, logs=None):
        one_cycle_dict = {'LearningRate': self.lrs, 'Momentum': self.moms}
        one_cycle_df = pd.DataFrame(one_cycle_dict)
        plot_path = os.path.join(ConfigObj.Path_Parent_Dir, 'src_train_model', 'plots_logs')
        one_cycle_csv_path = os.path.join(plot_path, "one_cycle_log.csv")
        one_cycle_df.to_csv(one_cycle_csv_path, index=False)
        print('Exit: on_train_end')
        #OneCycleScheduler.plot_once_cycle_frm_csv(one_cycle_csv_path)

    @staticmethod
    def plot_once_cycle_frm_csv(one_cycle_csv_path):
        plot_path = os.path.join(ConfigObj.Path_Parent_Dir, 'src_train_model', 'plots_logs')
        one_cycle_df = pd.read_csv(one_cycle_csv_path)
        ##accuracies
        lst_learning_rate = one_cycle_df['LearningRate'].tolist()
        lst_momentum = one_cycle_df['Momentum'].tolist()

        ax = plt.subplot(1, 2, 1)
        ax.plot(lst_learning_rate)
        ax.set_title('Learning Rate')
        ax = plt.subplot(1, 2, 2)
        ax.plot(lst_momentum)
        ax.set_title('Momentum')
        once_cycle_plot_path = os.path.join(plot_path, 'lr-momentum-steps.png')
        plt.savefig(once_cycle_plot_path)

##plot_hist_frm_csv('plots_logs/base_hist_log.csv', 'TRY')
#OneCycleScheduler.plot_once_cycle_frm_csv('/home/shunya/PythonProjects/Aptos2019/src_train_model/plots_logs/one_cycle_log.csv')