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

    The weights initialization and training can happen in 4 various formats:
    'R': Random weight initialization. Train all layers using standard learning rate
    'FA': Finetune all layers starting from a pretrained model
    'FL': Finetune last layer only using a pretrained model. Used as a set feature extractor
    'D': Differntiable learning. Use pretrained model and finetune all layers as per set diff learning rates

    Each model can be loaded as pretrained with Imagenet weights or not. Additionally, we offer the parameter
    to finetune all layers or not. If finetune_all_layers=True, then the model will be loaded and returned with all
    parameters of the model having requires_grad=True. However, if finetune_all_layers=False, we set the
    requires_grad for each model parameter to be False except for the newly added FC layer. The goal is to leverage
    pretrained weights fully and only modify the classifer on the end of the model.

    When using torchvision < 1.12 we may have to use a different format to load in models, as seen below.
    The preprocess for each model type can be found in the docs per each model
    model = torchvision.models.resnet18(pretrained=True)
    preprocess = torchvision.transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    """
    def __init__(self, model_name, class_names, weights_type='R'):
        super().__init__()
        self.class_names = class_names
        self.model_name = model_name
        self.diff_learning = True
        if weights_type == 'R': # Random weight initialization. Train all layers using standard learning rate
            self.pretrained = False
            self.finetune_all_layers = True
            self.diff_learning = False
        elif weights_type == 'FA': # Finetune all layers using a pretrained model
            self.pretrained = True
            self.finetune_all_layers = True
            self.diff_learning = False
        elif weights_type == 'FL': # Finetune last layer only using pretrained model
            self.pretrained = True
            self.finetune_all_layers = False
            self.diff_learning = False
        elif weights_type == 'D': # Differentiable Learning. Use pretrained model. Finetune all
            self.pretrained = True
            self.finetune_all_layers = True
            self.diff_learning = True

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

        param_to_update = []
        if self.diff_learning:
            param_to_update.append({"params": model.layer1.parameters(), "lr": 0.00001})
            param_to_update.append({"params": model.layer2.parameters(), "lr": 0.0001})
            param_to_update.append({"params": model.layer3.parameters(), "lr": 0.001})
            param_to_update.append({"params": model.layer4.parameters(), "lr": 0.001})
            param_to_update.append({"params": model.fc.parameters(), "lr": 0.001})
        else:
            for param in model.parameters():
                if param.requires_grad:
                    param_to_update.append(param)

        return model, preprocess, param_to_update

    def get_inceptionV3(self):
        weights = torchvision.models.Inception_V3_Weights.DEFAULT if self.pretrained else None
        preprocess = torchvision.models.Inception_V3_Weights.DEFAULT.transforms()
        model = torchvision.models.inception_v3(weights=weights)
        if not self.finetune_all_layers:
            for param in model.parameters():
                param.requires_grad = False
        # Handle the primary net
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(self.class_names))
        # Handle the auxiliary net
        num_features = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_features, len(self.class_names))

        param_to_update = []
        if self.diff_learning:
            param_to_update.append({"params": model.Mixed_5b.parameters(), "lr": 0.00001})
            param_to_update.append({"params": model.Mixed_5c.parameters(), "lr": 0.00001})
            param_to_update.append({"params": model.Mixed_5d.parameters(), "lr": 0.00001})
            param_to_update.append({"params": model.Mixed_6a.parameters(), "lr": 0.0001})
            param_to_update.append({"params": model.Mixed_6b.parameters(), "lr": 0.0001})
            param_to_update.append({"params": model.Mixed_6c.parameters(), "lr": 0.001})
            param_to_update.append({"params": model.Mixed_6d.parameters(), "lr": 0.001})
            param_to_update.append({"params": model.Mixed_6e.parameters(), "lr": 0.001})
            param_to_update.append({"params": model.AuxLogits.parameters(), "lr": 0.001})
            param_to_update.append({"params": model.Mixed_7a.parameters(), "lr": 0.001})
            param_to_update.append({"params": model.Mixed_7b.parameters(), "lr": 0.001})
            param_to_update.append({"params": model.Mixed_7c.parameters(), "lr": 0.001})
            param_to_update.append({"params": model.fc.parameters(), "lr": 0.001})
        else:
            for param in model.parameters():
                if param.requires_grad:
                    param_to_update.append(param)

        return model, preprocess, param_to_update

    def get_vgg16(self):
        weights = torchvision.models.VGG16_Weights.DEFAULT if self.pretrained else None
        preprocess = torchvision.models.VGG16_Weights.DEFAULT.transforms()
        model = torchvision.models.vgg16(weights=weights)
        if not self.finetune_all_layers:
            for param in model.features.parameters():
                param.requires_grad = False
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, len(self.class_names))

        param_to_update = []
        if self.diff_learning:
            param_to_update.append({"params": [model.features[10].parameters(),
                                               model.features[12].parameters(),
                                               model.features[14].parameters()], "lr": 0.00001})
            param_to_update.append({"params": [model.features[17].parameters(),
                                               model.features[19].parameters(),
                                               model.features[21].parameters()], "lr": 0.0001})
            param_to_update.append({"params": [model.features[24].parameters(),
                                               model.features[26].parameters(),
                                               model.features[28].parameters()], "lr": 0.001})
            param_to_update.append({"params": model.classifier.parameters(), "lr": 0.001})
        else:
            for param in model.parameters():
                if param.requires_grad:
                    param_to_update.append(param)

        return model, preprocess, param_to_update

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

        param_to_update = []
        if self.diff_learning:
            param_to_update.append({"params": [model.features[6].parameters(),
                                               model.features[7].parameters()], "lr": 0.00001})
            param_to_update.append({"params": [model.features[9].parameters(),
                                               model.features[10].parameters()], "lr": 0.0001})
            param_to_update.append({"params": [model.features[11].parameters(),
                                               model.features[12].parameters()], "lr": 0.001})
            param_to_update.append({"params": model.classifier.parameters(), "lr": 0.001})
        else:
            for param in model.parameters():
                if param.requires_grad:
                    param_to_update.append(param)

        return model, preprocess, param_to_update
