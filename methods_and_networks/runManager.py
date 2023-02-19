"""
Author: Devesh Shah
Project Title: Pytorch Base Format

This file includes several helper functions we use to coordinate our training.
In particular we have a Network Factory, an Optimizer Factory and a Run Manager which handles
the epoch/run start/end commands. The Run Manager is our primary source of coordinating the tracking
of hyper-parameters and training results.
"""


import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict, namedtuple
from itertools import product
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import io
from methods_and_networks.networks import *
from methods_and_networks.helperFxns import *


class NetworkFactory:
    """
    The NetworkFactory loads in various types of predefined models from the networks.py file.
    This can be customized to generate various options of models using different parameters
    """
    @staticmethod
    def get_network(name):
        if name == "Network1":
            return Network1()
        elif name == "Network2":
            return Network2()
        elif name == "Network2_DO":
            return Network2(conv_dropout=0.2, fc_dropout=0.5)
        elif name == "Network2withBN":
            return Network2withBN(conv_dropout=0.2, fc_dropout=0.5)
        else:
            raise Exception("Invalid Network Name")


class OptimizerFactory:
    """
    The OptimizerFactory loads in various types of optimizers that are provided by torch.optim.
    Many parameters are available for the user to easily modify the hyper-parameters of the optimizer
    """
    @staticmethod
    def get_optimizer(name, params, lr, momentum=0, dampening=0, betas=(0.9, 0.99), eps=1e-08, weight_decay=0, alpha=0.99):
        if name == "Adam":
            optimizer = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        elif name == "SGD":
            optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)
        elif name == "RMSprop":
            optimizer = torch.optim.RMSprop(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)
        else:
            raise Exception("Invalid Optimizer Type Specified")
        return optimizer


class LRSchedulerFactory:
    """
    The LR Scheduler Factory loads in various types of schedulers that are provided by torch.optim. We have currently
    only defined 4 different options: None (use constant LR), CosineAnnealingLR, CosineAnnealingWarmRestarts, and
    ReduceLROnPlateau. There are plenty of other options given by torch.optim not defined here.

    There exists plenty of documentation on the different methods:
    https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863
    """
    @staticmethod
    def get_lr_scheduler(name, optimizer, epochs, eta_min=0.00001, t_max=None, t_0=None, t_mult=None):
        if name == "None":
            const_lambda = lambda epoch: 1 ** epoch
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=const_lambda)
        elif name == "Cosine":
            if t_max is None: scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
            else: scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        elif name == "CosineWithRestarts":
            if t_0 is None or t_mult is None: scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(int(epochs/4), 1), T_mult=1, eta_min=eta_min)
            else: scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min)
        elif name == "ReduceOnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        else:
            raise Exception("Invalid LR Scheduler Type Specified")
        return scheduler


class RunBuilder:
    """
    The RunBuilder uses the product method to find all possible combinations of hyper-parameters we
    hope to run in the model
    """
    @staticmethod
    def get_runs(params):
        RunParams = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(RunParams(*v))
        return runs


class RunManager:
    """
    The RunManager is our centralized class that helps manage and track each epoch/run in an organized
    fashion. This has been adapated from https://deeplizard.com/

    In particular, this class performs all the addition to tensorboard and calculates loss/accuracy
    after each trial, keeping track of all parameters. All the final results are saved to a dataframe
    for further study after the training is complete
    """
    def __init__(self, device, use_tensorboard, score_by):
        self.device = device
        self.epoch = Epoch()
        self.run = Run()

        self.network = None
        self.hyper_parameters = None
        self.loader = None
        self.val_loader = None
        self.confusion_matrix = None
        self.class_labels = None
        self.tb = None
        self.use_tb = use_tensorboard
        self.score_by_acc = score_by['Accuracy']
        self.min_loss = float('inf')

    def begin_run(self, run, network, loader, val_loader, hyparams, class_labels):
        self.run.new_run(run)
        self.hyper_parameters = hyparams
        self.class_labels = class_labels
        self.network = network
        self.loader = loader
        self.val_loader = val_loader

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        if self.use_tb:
            self.tb = SummaryWriter(comment=f'-Run{self.run.count}_{run}')  # for windows OS - too long of a file name
            self.tb.add_image('Initial_Images', grid)

            # Add graph to Tensorboard consumes large amount of system memory (RAM) over many runs
            # and accumulates. Avoid using this when possible. However, this can be useful to compare model
            # architectures visually when working with only a few examples.
            # self.tb.add_graph(self.network.to('cpu'), images)
            # self.network.to(self.device)

    def end_run(self):
        # Add hyper-parameters for study
        hyper_param_dict = OrderedDict()
        for idx, hyper_param in enumerate(self.run.params):
            hyper_param_dict[self.hyper_parameters[idx]] = hyper_param

        if self.use_tb:
            self.tb.add_hparams(dict(hyper_param_dict),
                                {'train_loss': self.epoch.loss / len(self.loader.dataset),
                                 'val_loss': self.epoch.val_loss / len(self.val_loader.dataset),
                                 'train_acc': self.epoch.num_correct / len(self.loader.dataset),
                                 'val_acc': self.epoch.val_num_correct / len(self.val_loader.dataset)})
            self.tb.close()

        self.epoch.reset()
        self.save('training_results')  # Accumulates RAM over run. Comment out if only need results at end.

    def begin_epoch(self):
        self.epoch.new_epoch()

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch.start_time
        run_duration = time.time() - self.run.start_time

        loss = self.epoch.loss / len(self.loader.dataset)
        val_loss = self.epoch.val_loss / len(self.val_loader.dataset)
        accuracy = self.epoch.num_correct / len(self.loader.dataset)
        val_accuracy = self.epoch.val_num_correct / len(self.val_loader.dataset)

        if self.use_tb:
            self.tb.add_scalar('Loss', loss, self.epoch.count)
            self.tb.add_scalar('Val_Loss', val_loss, self.epoch.count)
            self.tb.add_scalar('Accuracy', accuracy, self.epoch.count)
            self.tb.add_scalar('Val_Accuracy', val_accuracy, self.epoch.count)

            # Add histogram to Tensorboard consumes large amount of system memory (RAM) over many runs
            # and accumulates. Avoid using this when possible. However, this can be useful to make sure
            # training of weights is progressing properly.
            # for name, param in self.network.named_parameters():
            #     self.tb.add_histogram(name, param, self.epoch.count)
            #     self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch.count)

            # For every 5 epochs, print a confusion matrix
            if self.epoch.count % 5 == 0:
                self._get_confusion_mat()
                figure = plt.figure(figsize=(8, 8))
                sns.heatmap(self.confusion_matrix / np.sum(self.confusion_matrix), annot=True, fmt='.1%', cmap='Blues',
                            cbar=True, xticklabels=self.class_labels, yticklabels=self.class_labels)
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label' + f'\n\nValidation Accuracy={val_accuracy}')
                plt.title('Confusion Matrix Validation Dataset')
                plt.tight_layout()
                self.tb.add_figure("Confusion Matrix", figure, self.epoch.count)

        results = OrderedDict()
        results['run'] = self.run.count
        results['epoch'] = self.epoch.count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['val_loss'] = val_loss
        results['val_accuracy'] = val_accuracy
        results['epoch_duration'] = epoch_duration
        results['run_duration'] = run_duration
        results['learning_rate'] = self.epoch.sched_lr

        for k, v in self.run.params._asdict().items():
            results[k] = v

        self.run.data.append(results)

        # Uncomment if would like to display df with all epoch results after each epoch
        # Best used when working with a jupyter notebook
        # df = pd.DataFrame.from_dict(self.run.data, orient='columns')
        # clear_output(wait=True)
        # display(df)

    def track_loss(self, loss, data_type):
        if data_type == 'train':
            self.epoch.loss += loss.item() * self.loader.batch_size
        elif data_type == 'val':
            self.epoch.val_loss += loss.item() * self.val_loader.batch_size
        else:
            raise Exception("Invalid Loss Tracker Type")

    def track_num_correct(self, preds, labels, data_type):
        if data_type == 'train':
            self.epoch.num_correct += self._get_num_correct(preds, labels)
        elif data_type == 'val':
            self.epoch.val_num_correct += self._get_num_correct(preds, labels)
        else:
            raise Exception("Invalid Accuracy Tracker Type")

    def track_sched_lr(self, lr):
        self.epoch.sched_lr = lr

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    @torch.no_grad()
    def _get_confusion_mat(self):
        self.network.eval()
        all_preds = torch.Tensor()
        all_labels = torch.Tensor()
        for batch in self.val_loader:
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            all_labels = torch.cat([all_labels, labels.to('cpu')], dim=0)
            preds = self.network(images)
            preds = preds.argmax(dim=1)
            all_preds = torch.cat([all_preds, preds.to('cpu')], dim=0)
        self.confusion_matrix = confusion_matrix(all_labels, all_preds)

    def save(self, filename):
        fig, lgd, self.run_names = helperFxn_plot_topK(self.run.data, self.run.names, self.score_by_acc)
        # The xlsxwriter accumulates memory as it is called after each end_run. This is a known issue with functionality
        # https://xlsxwriter.readthedocs.io/working_with_memory.html
        writer = pd.ExcelWriter(filename + '.xlsx', engine='xlsxwriter')
        df = pd.DataFrame.from_dict(self.run.data, orient='columns').to_excel(writer, sheet_name='All_Data')
        run_names_ordered = pd.DataFrame.from_dict(self.run.names, orient='columns').copy(deep=True)
        if self.score_by_acc:
            run_names_ordered = run_names_ordered.sort_values(by=['Best_Performance'], ascending=False)
        else:
            run_names_ordered = run_names_ordered.sort_values(by=['Best_Performance'], ascending=True)
        df_temp = pd.DataFrame().reindex_like(run_names_ordered).dropna().iloc[:, 0:2]
        run_names_toprint = pd.concat([run_names_ordered.reset_index(drop=True), df_temp.reset_index(drop=True), pd.DataFrame.from_dict(self.run.names, orient='columns').reset_index(drop=True)], axis=1, ignore_index=True)
        run_names_toprint.columns = ['Run_Number', 'Best_Performance', 'Run_Name', '-', '-', 'Run_Number', 'Best_Performance', 'Run_Name']
        df2 = run_names_toprint.to_excel(writer, sheet_name='All_Plots')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        buf.seek(0)
        worksheet = writer.sheets['All_Plots']
        worksheet.insert_image('C' + str(self.run.count + 5), 'Accuracy Plots', options={'image_data': buf})
        writer.save()
        buf.close()
