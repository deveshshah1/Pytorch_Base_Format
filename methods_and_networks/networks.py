"""
Author: Devesh Shah
Project Title: Pytorch Base Format

This file represents the various networks we will use for training. We implement several state of the art models
here for reference. We offer the option for using either pretrained or randomly initialized weights.

Different model architectures can be leveraged based on the problem we are working with.
"""


import torch.nn as nn
import torch.nn.functional as F
import torchvision


class TransferLearningNetworks():
    """
    This class generates various predefined models from pytorch. These include Resnet18, Resnet50, Resnet101,
    InceptionV3, VGG16, and SqueezeNet.

    For each model, we change the final fc layer or the conv layer to output the number of classes we have in our
    problem domain.

    Each model can be loaded as pretrained with Imagenet weights or not. Additionally, we offer the parameter
    to finetune all layers or not. If finetune_all_layers=True, then the model will be loaded and returned with all
    parameters of the model having requires_grad=True. However, if finetune_all_layers=False, we set the
    requires_grad for each model parameter to be False except for the newly added FC layer. The goal is to leverage
    pretrained weights fully and only modify the classifer on the end of the model.
    """
    def __init__(self, model_name, class_names, pretrained=True, finetune_all_layers=False):
        super().__init__()
        self.class_names = class_names
        self.pretrained = pretrained
        self.model_name = model_name
        self.finetune_all_layers = finetune_all_layers

    def get_model(self):
        if self.model_name == 'resnet18' or self.model_name == 'resnet50' or self.model_name == 'resnet101':
            return self.get_resnet(self.model_name)
        elif self.model_name == 'inceptionV3':
            return self.get_inceptionV3()
        elif self.model_name == 'vgg16':
            return self.get_vgg16()
        elif self.model_name == 'squeezenet':
            return self.get_squeezenet()
        else:
            raise Exception('Invalid Network Name')

    def get_resnet(self, model_size):
        if model_size == 'resnet18':
            weights = torchvision.models.ResNet18_Weights.DEFAULT if self.pretrained else None
            preprocess = torchvision.models.ResNet18_Weights.DEFAULT.transforms()
            model = torchvision.models.resnet18(weights=weights)
        elif model_size == 'resnet50':
            weights = torchvision.models.ResNet50_Weights.DEFAULT if self.pretrained else None
            preprocess = torchvision.models.ResNet50_Weights.DEFAULT.transforms()
            model = torchvision.models.resnet50(weights=weights)
        elif model_size == 'resnet101':
            weights = torchvision.models.ResNet101_Weights.DEFAULT if self.pretrained else None
            preprocess = torchvision.models.ResNet101_Weights.DEFAULT.transforms()
            model = torchvision.models.resnet101(weights=weights)

        if not self.finetune_all_layers:
            for param in model.parameters():
                param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(self.class_names))
        return model, preprocess

    def get_inceptionV3(self):
        weights = torchvision.models.Inception_V3_Weights.DEFAULT if self.pretrained else None
        preprocess = torchvision.models.Inception_V3_Weights.DEFAULT.transforms()
        model = torchvision.models.inception_v3(weights=weights)
        if not self.finetune_all_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(self.class_names))
        return model, preprocess

    def get_vgg16(self):
        weights = torchvision.models.VGG16_Weights.DEFAULT if self.pretrained else None
        preprocess = torchvision.models.VGG16_Weights.DEFAULT.transforms()
        model = torchvision.models.vgg16(weights=weights)
        if not self.finetune_all_layers:
            for param in model.features.parameters():
                param.requires_grad = False
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, len(self.class_names))
        return model, preprocess

    def get_squeezenet(self):
        weights = torchvision.models.SqueezeNet1_1_Weights.DEFAULT if self.pretrained else None
        preprocess = torchvision.models.SqueezeNet1_1_Weights.DEFAULT.transforms()
        model = torchvision.models.squeezenet1_1(weights=weights)
        if not self.finetune_all_layers:
            for param in model.features.parameters():
                param.requires_grad = False
        num_channels = model.classifier[1].in_channels
        model.classifier[1] = nn.Conv2d(num_channels, len(self.class_names), kernel_size=(1, 1))
        model.num_classes = len(self.class_names)
        return model, preprocess
