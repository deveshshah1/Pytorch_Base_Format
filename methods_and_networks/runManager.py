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


class NetworkFactory():
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


class OptimizerFactory():
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


class RunBuilder():
    """
    The RunBuilder uses the product method to find all possible combinations of hyper-parameters we
    hope to run in the model
    """
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs


class RunManager():
    """
    The RunManager is our centralized class that helps manage and track each epoch/run in an organized
    fashion. This has been adapated from https://deeplizard.com/

    In particular, this class performs all the addition to tensorboard and calculates loss/accuracy
    after each trial, keeping track of all parameters. All the final results are saved to a dataframe
    for further study after the training is complete
    """
    def __init__(self, device):
        self.device = device

        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_val_loss = 0
        self.epoch_num_correct = 0
        self.epoch_val_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_names = {'Run_Number': [], 'Run_Name': []}
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.hyper_parameters = None
        self.loader = None
        self.val_loader = None
        self.confusion_matrix = None
        self.class_labels = None
        self.tb = None

    def begin_run(self, run, network, loader, val_loader, hyparams, class_labels):
        self.run_start_time = time.time()
        self.run_params = run
        self.hyper_parameters = hyparams
        self.class_labels = class_labels
        self.run_count += 1
        self.network = network
        self.loader = loader
        self.val_loader = val_loader
        self.run_names['Run_Number'].append(self.run_count)
        self.run_names['Run_Name'].append(f'{run}')
        self.tb = SummaryWriter(comment=f'-{run}')  # note for windows OS, run may be too long of a file name

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)
        self.tb.add_image('Initial_Images', grid)
        # Add graph to Tensorboard consumes large amount of system memory (RAM) over many runs
        # and accumulates. Avoid using this when possible. However, this can be useful to compare model
        # architectures visually when working with only a few examples.
        # self.tb.add_graph(self.network.to('cpu'), images)
        # self.network.to(self.device)

    def end_run(self):
        # Add hyper-parameters for study
        hyper_param_dict = OrderedDict()
        for idx, hyper_param in enumerate(self.run_params):
            hyper_param_dict[self.hyper_parameters[idx]] = hyper_param
        self.tb.add_hparams(dict(hyper_param_dict),
                            {'train_loss': self.epoch_loss / len(self.loader.dataset),
                             'val_loss': self.epoch_val_loss / len(self.val_loader.dataset),
                             'train_acc': self.epoch_num_correct / len(self.loader.dataset),
                             'val_acc': self.epoch_val_num_correct / len(self.val_loader.dataset)})

        self.tb.close()
        self.epoch_count = 0
        self.save('training_results')  # Accumulates RAM over run. Comment out if only need results at end.

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_val_loss = 0
        self.epoch_num_correct = 0
        self.epoch_val_num_correct = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        val_loss = self.epoch_val_loss / len(self.val_loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)
        val_accuracy = self.epoch_val_num_correct / len(self.val_loader.dataset)

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Val_Loss', val_loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
        self.tb.add_scalar('Val_Accuracy', val_accuracy, self.epoch_count)

        # Add histogram to Tensorboard consumes large amount of system memory (RAM) over many runs
        # and accumulates. Avoid using this when possible. However, this can be useful to make sure
        # training of weights is progressing properly.
        # for name, param in self.network.named_parameters():
        #     self.tb.add_histogram(name, param, self.epoch_count)
        #     self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        # For every 5 epochs, print a confusion matrix
        if self.epoch_count % 5 == 0:
            self._get_confusion_mat()
            figure = plt.figure(figsize=(8, 8))
            sns.heatmap(self.confusion_matrix / np.sum(self.confusion_matrix), annot=True, fmt='.1%', cmap='Blues',
                        cbar=True, xticklabels=self.class_labels, yticklabels=self.class_labels)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label' + f'\n\nValidation Accuracy={val_accuracy}')
            plt.title('Confusion Matrix Validation Dataset')
            plt.tight_layout()
            self.tb.add_figure("Confusion Matrix", figure, self.epoch_count)

        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['val_loss'] = val_loss
        results['val_accuracy'] = val_accuracy
        results['epoch_duration'] = epoch_duration
        results['run_duration'] = run_duration

        for k, v in self.run_params._asdict().items():
            results[k] = v

        self.run_data.append(results)

        # Uncomment if would like to display df with all epoch results after each epoch
        # Best used when working with a jupyter notebook
        # df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        # clear_output(wait=True)
        # display(df)

    def track_loss(self, loss, data_type):
        if data_type == 'train':
            self.epoch_loss += loss.item() * self.loader.batch_size
        elif data_type == 'val':
            self.epoch_val_loss += loss.item() * self.val_loader.batch_size
        else:
            raise Exception("Invalid Loss Tracker Type")

    def track_num_correct(self, preds, labels, data_type):
        if data_type == 'train':
            self.epoch_num_correct += self._get_num_correct(preds, labels)
        elif data_type == 'val':
            self.epoch_val_num_correct += self._get_num_correct(preds, labels)
        else:
            raise Exception("Invalid Accuracy Tracker Type")

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
        fig, lgd = helperFxn_plot_all_accuracy(self.run_data)
        # The xlsxwriter accumulates memory as it is called after each end_run. This is a known issue with functionality
        # https://xlsxwriter.readthedocs.io/working_with_memory.html
        writer = pd.ExcelWriter(filename + '.xlsx', engine='xlsxwriter')
        df = pd.DataFrame.from_dict(self.run_data, orient='columns').to_excel(writer, sheet_name='All_Data')
        df2 = pd.DataFrame.from_dict(self.run_names, orient='columns').to_excel(writer, sheet_name='All_Plots')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        buf.seek(0)
        worksheet = writer.sheets['All_Plots']
        worksheet.insert_image('C' + str(self.run_count+ 5), 'Accuracy Plots', options={'image_data': buf})
        writer.save()
        buf.close()
