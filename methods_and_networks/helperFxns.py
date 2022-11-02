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
import math


def helperFxn_plot_topK(run_data, run_names, score_by_acc, k=5):
    """
    This function plots the validation accuracy or validation loss for all the runs across all the epochs onto one
    graph and returns the figure. Additionally, this fxn finds the k=5 highest validation accuracy's or k=5 lowest
    validation losses across all the runs and highlights these runs on the graph to indicate the run number using
    a thicker line.

    This fxn can easily be adapted to plot other properties between all runs and how many top K we would like to
    emphasize.
    :param run_data: A list structure of the results from every epoch for every run. Comes from the RunManager
    variable self.run_data
    :param run_names: A list of run names along with run number. Comes from RunManager variable self.run_names
    :param score_by_acc: If True, find max validation accuracies. If False, find min validation losses
    :return: A figure with the validation accuracy of every run plotted on one graph. Top K val accuracy runs are
    specially highlighted. Also return run_names with best performance filled out
    """
    df = pd.DataFrame.from_dict(run_data, orient='columns')
    run_values_dict = {}
    run_list, ordered_val_list = [], []

    for index, row in df.iterrows():
        run_number = row['run']
        if run_number in run_values_dict:
            if score_by_acc:
                run_values_dict[run_number].append(row['val_accuracy'])
            else:
                run_values_dict[run_number].append(row['val_loss'])
        else:
            if score_by_acc:
                run_values_dict[run_number] = [row['val_accuracy']]
            else:
                run_values_dict[run_number] = [row['val_loss']]

    for key, value in run_values_dict.items():
        if score_by_acc:
            ordered_val_list.append(max(value))
        else:
            ordered_val_list.append(min(value))
        run_list.append(key)

    run_names['Best_Performance'] = ordered_val_list
    ordered_val_list, run_list = (list(t) for t in zip(*sorted(zip(ordered_val_list, run_list))))
    if score_by_acc:
        run_list = run_list[-k:]
    else:
        run_list = run_list[:k]

    fig, ax = plt.subplots()
    for key, value in run_values_dict.items():
        if key in run_list:
            ax.plot(np.arange(1, len(value) + 1), value, label=key, linewidth=4)
        else:
            ax.plot(np.arange(1, len(value) + 1), value, linestyle='dashed', label=key)

    if score_by_acc:
        ax.set_ylabel('Validation_Accuracy')
    else:
        ax.set_ylabel('Validation_Loss')
    ax.set_xlabel('Epoch')
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 1.1, pos.height])
    ncols = min(math.ceil(len(ordered_val_list) / 25.0), 4)
    anchor_sizes = {1: 1.25, 2: 1.35, 3: 1.5, 4: 1.65}
    lgd = ax.legend(loc='center right', bbox_to_anchor=(anchor_sizes[ncols], 0.5), ncol=ncols)
    return fig, lgd, run_names
