"""
Author: Devesh Shah
Project Title: Pytorch Base Format

The main purpose of this file is to serve as a dictionary of methods that can be taken and adapted for particular
use cases as desired.

This file includes several helper functions that can be useful for various different methods. This file, though
not directly used by main.py during training, is a compilation of methods and their implementations. They
can be taken from here and modified in order to use for training/eval.

In particular, we highlight the following methods:
- Data-loader Transforms with augmentation for training
- Feature Extractors using Pytorch Hooks
- Differential Learning
- Visualize Sample Predictions of Trained Model
- Custom Data Loader
"""


import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import json
import os
import sklearn.preprocessing as skl


class CustomDataLoader(torch.utils.data.Dataset):
    """
    This is an example of a custom dataloader using Torch Datasets. They key components include the __len__ and the
    __getitem__ functions. These are required to over-ride the dataloaders. The exact format will depend on how the
    images and csv data is organized. Different forms of standardization/normalization are implemented here.
    It may also be required to create custom transformations based on the image format. See example below:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(self, root_dir, csv_dir, train_test_names, targets, transform=None, train=True, scaling=None):
        self.root_dir = root_dir
        self.transform = transform
        self.targets = targets
        self.df_orig = pd.read_csv(csv_dir)
        self.df = self.df_orig.copy(deep=True)

        train_test_split = json.load(open(train_test_names))
        self.img_names = train_test_split['full_train'] if train else train_test_names['full_test']

        self.scalars = {}
        if scaling == 'Normalization':
            for target in targets:
                self.scalars[target] = skl.MinMaxScaler().fit(pd.DataFrame(self.df_orig[target]))
                self.df[target] = pd.DataFrame(self.scalars[target].transform(pd.DataFrame(self.df_orig[target])))
        elif scaling == 'Standardization':
            for target in targets:
                self.scalars[target] = skl.StandardScaler().fit(pd.DataFrame(self.df_orig[target]))
                self.df[target] = pd.DataFrame(self.scalars[target].transform(pd.DataFrame(self.df_orig[target])))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_name = self.img_names[item]
        img_df = self.df.loc[self.df['name'] == img_name]
        img_path = os.path.join(self.root_dir, img_name + '.png')
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        labels = []
        for target in self.targets:
            labels.append(img_df.iloc[0][target])
        labels = torch.Tensor(labels)
        return image, labels

    def inverse_scaling(self, item, target):
        """
        If you perform scaling (normalization/standardization), you will need to undo that transform in order
        to obtain the true prediction that outputted from a model when it is being used.
        This is one example of undo-ing that operation
        """
        return self.scalars[target].inverse_transform(item)


def get_transform_with_aug():
    """
    This provides an example of using augmentations from the transforms class when creating the data-loader.
    The key point is that we don't apply the same augmenations to the validation set. We only want to perform
    the same normalization and resize for the val images, but the images should not be perturbed as we are not
    aiming to increase generalization ability, but score on these real images as they are.

    Additional transforms can be found at:
    https://pytorch.org/vision/stable/transforms.html

    :return: data_transforms, a dictionary of transforms for train and val
    """
    input_size = None
    data_transforms = {
        'train': transforms.Compose([transforms.RandomResizedCrope(input_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(10),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, ), (0.5, ))]),
        'val': transforms.Compose([transforms.Resize(input_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, ), (0.5, ))])
    }
    return data_transforms


def feature_extractor():
    """
    Feature extractors are very commonly used in deep learning. We demonstrate how to use pytorch hooks to extract
    features from a pretrained model. This can be modified for any layer/any number of layers.

    Inspired from: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254
    :return: features of given image
    """
    activation = {}
    def get_activation(name):
        # The name parameter is flexible. This has no impact on what layer is extracted. It is a way to
        # save the output into a dictionary for access later
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model = torchvision.models.inception_v3(pretrained=True)
    # alternately create a custom network and load set of pretrained weights from a pth file
    # model.XXX is where we input what layer we hope to extract features from
    model.avgpool.register_forward_hook(get_activation('avgpool'))
    model.eval()

    tforms = get_transform_with_aug()
    all_features = torch.Tensor()
    all_labels = []
    listOfImages = []

    for image in listOfImages:
        activation = {}
        img = Image.open(image).convert('RGB')
        model_in = tforms['val'](img)
        model_in = model_in.unsqueeze(dim=0)
        model_out = model(model_in)

        img_features = activation['avgpool']
        img_features = img_features.squeeze(dim=3).squeeze(dim=2)
        all_features = torch.cat((all_features, img_features), dim=0)
        all_labels.append(image)

    all_features = all_features.cpu().detach().numpy()

    return all_features, all_labels


def differential_learning():
    """
    Differntiale learning is the means to apply different learning rates to different layers of the network during
    trianing. In particular, this is useful when working with transfer learning models and using pretrained weights.
    Diff Learning allows us to use smaller learning rates near the start of the network and larger rates near the
    later layers, thus performing more fine tuning on later layers for detailed anaylsis.

    Additionally, we can apply a similar method to freeze learning of a few layers (usually the first few) and only
    update the later layers.

    Further Notes: https://stackoverflow.com/questions/51801648/how-to-apply-layer-wise-learning-rate-in-pytorch
    :return: optimizer with differentiable learning rate applied
    """
    model = torchvision.models.inception_v3(pretrained=True)

    # print model to see different layers and choose param/lr combination based on that
    params_to_update = []
    params_to_update.append({"params": model.layer1.parameters(), "lr": 0.00001})
    params_to_update.append({"params": model.layer2.parameters(), "lr": 0.0001})
    params_to_update.append({"params": model.layer3.parameters(), "lr": 0.001})
    params_to_update.append({"params": model.layer4.parameters(), "lr": 0.001})
    params_to_update.append({"params": model.fc.parameters()})
    # Only parameters included in optim will update. For any params in params_to_update without a lr, will use global
    # lr that is included in optimizer initialization
    # If you have
    optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    return optimizer


def visualize_sample_predictions(dataloader_val, device, class_names):
    """
    This method is used to visualize several predictions using a trained model. The predictions are moade on
    images from the validation set and then output to the user showing the image, prediction, and true label.

    This can be modified to save these predictions every so many epochs to understand failed states
    :return: visualization of predictions of trained model on images
    """
    model = torchvision.models.inception_v3(pretrained=True)
    was_training = model.training
    model.eval()
    num_images = 6
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader_val):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for j in range(images.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]} \n actual: {class_names[labels[j]]}')
                plt.imshow(images.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

        model.train(mode=was_training)
