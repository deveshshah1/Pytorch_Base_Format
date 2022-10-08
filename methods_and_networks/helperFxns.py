"""
Author: Devesh Shah
Project Title: Pytorch Base Format

This file holds all the suplementary helper functions we are currently using in the Run Manager / main loop. The
functions in this file are separated here to create clarity when reading the other methods files.

All the functions in this file start with the naming structure "helperFxn_..." to help recognize where the
implementation of these methods can be found.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def helperFxn_plot_all_accuracy(run_data):
    """
    This function plots the validation accuracy for all the runs across all the epochs onto one graph and returns
    the figure. Additionally, this fxn finds the 5 highest validation accuracy's across all the runs and highlights
    these runs on the graph to indicate the run number using a thicker line.

    This fxn can easily be adapted to plot other properties between all runs and how many top K we would like to
    emphasize.
    :param run_data: A list structure of the results from every epoch for every run. Comes from the RunManager
    variable self.run_data
    :return: A figure with the validation accuracy of every run plotted on one graph. Top K val accuracy runs are
    specially highlighted
    """
    df = pd.DataFrame.from_dict(run_data, orient='columns')
    run_val_loss_dict = {}
    run_list, max_val_acc_list = [], []

    for index, row in df.iterrows():
        run_number = row['run']
        if run_number in run_val_loss_dict:
            run_val_loss_dict[run_number].append(row['val_accuracy'])
        else:
            run_val_loss_dict[run_number] = [row['val_accuracy']]

    for key, value in run_val_loss_dict.items():
        max_val_acc_list.append(max(value))
        run_list.append(key)

    max_val_acc_list, run_list = (list(t) for t in zip(*sorted(zip(max_val_acc_list, run_list))))
    run_list = run_list[-5:]

    fig, ax = plt.subplots()
    for key, value in run_val_loss_dict.items():
        if key in run_list:
            ax.plot(np.arange(1, len(value) + 1), value, label=key, linewidth=4)
        else:
            ax.plot(np.arange(1, len(value) + 1), value, linestyle='dashed', label=key)

    ax.set_ylabel('Validation_Accuracy')
    ax.set_xlabel('Epoch')
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width *0.9, pos.height])
    ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    return fig