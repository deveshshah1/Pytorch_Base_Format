"""
Author: Devesh Shah
Project Title: Pytorch Base Format

This file includes several helper functions that can be useful for various different methods. This file, though
not directly used by main.py during training, is a compilation of methods and their implementations. They
can be taken from here and modified in order to use for training/eval.

In particular, we highlight the following methods:
- Data-loader Transforms with augmentation for training
- Feature Extractors using Pytorch Hooks
"""


import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image


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
