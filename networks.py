"""
Author: Devesh Shah
Project Title: Pytorch Base Format

This file represents the various networks we will use for training. We implement several simplistic models here
for reference. We include properties such as Dropout and Batch Norm in order to show improvements from basic models.

For most trainings in the future you will leverage much deeper models. Consider transfer learning methods as well
for well known model architectures.
"""


import torch.nn as nn
import torch.nn.functional as F


class Network1(nn.Module):
    """
    Network1 is the most basic architecture with 2 conv layers and 2 fc layers
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=(12*5*5), out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        t = t

        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 12*5*5)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)

        return t


class Network2(nn.Module):
    """
    Network2 is a deeper CNN architecture. It also includes the opportunity to include dropout given the
    correct parameters to the init function
    """
    def __init__(self, conv_dropout=0, fc_dropout=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=(1, 1))
        self.conv4_bn = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=(1, 1))
        self.conv6_bn = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(in_features=(128 * 4 * 4), out_features=256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.out = nn.Linear(in_features=64, out_features=10)

        self.conv_dropout = nn.Dropout(conv_dropout)
        self.fc_dropout = nn.Dropout(fc_dropout)

    def forward(self, t):
        t = t

        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2)
        t = self.conv_dropout(t)

        t = F.relu(self.conv3(t))
        t = F.relu(self.conv4(t))
        t = F.max_pool2d(t, kernel_size=2)
        t = self.conv_dropout(t)

        t = F.relu(self.conv5(t))
        t = F.relu(self.conv6(t))
        t = F.max_pool2d(t, kernel_size=2)
        t = self.conv_dropout(t)

        t = t.reshape(-1, 128*4*4)
        t = F.relu(self.fc1(t))
        t = self.fc_dropout(t)
        t = F.relu(self.fc2(t))
        t = self.fc_dropout(t)

        t = self.out(t)

        return t


class Network2withBN(nn.Module):
    """
    Network2withBN is a similar architecture to Network2 with Batch Norm introduced. Additionally, we leverage
    the nn.Sequential method to highlight differet ways of building networks when repeating blocks exist.
    """
    def __init__(self, conv_dropout=0, fc_dropout=0):
        super().__init__()
        self.conv_model = nn.Sequential(
            self.conv_building_block(in_chan=3, out_chan=32, dropout=conv_dropout),
            self.conv_building_block(in_chan=32, out_chan=64, dropout=conv_dropout),
            self.conv_building_block(in_chan=64, out_chan=128, dropout=conv_dropout)
        )

        self.fc_model = nn.Sequential(
            self.fc_building_block((128*4*4), 256, dropout=fc_dropout),
            self.fc_building_block(256, 64, dropout=fc_dropout),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, t):
        t = self.conv_model(t)
        t = t.reshape(-1, 128*4*4)
        t = self.fc_model(t)

        return t

    def conv_building_block(self, in_chan, out_chan, kernel_size=3, padding=(1, 1), dropout=0):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout)
        )
        return block

    def fc_building_block(self, in_feat, out_feat, dropout=0):
        block = nn.Sequential(
            nn.Linear(in_features=in_feat, out_features=out_feat),
            nn.BatchNorm1d(out_feat),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        return block
