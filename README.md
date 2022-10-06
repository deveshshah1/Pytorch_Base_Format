# Pytorch_Base_Format
Many pytorch models can be trained in a similar fashion. This repo includes a baseline format that can be used in many different pytorch models. The goal is to create a standard format to begin new pytorch projects. In particular, we modify the RunManager and RunBuilder class structure provided by DeepLizard [1]. This structure allows us to organize the program into simple classes which can be easily interpretted. This repo is a basic scenario of model development and needs to be modified for more complex scenarios. 

The main branch of this repo includes a training for a simple CNN model. Each additional branch represents templates for various alternative models that can be trained. We include capabilities for the following: 
- Convoluation Neural Network (CNN): Main Branch
- Transfer Learning
- AutoEncoder (AE)
- Variational AutoEncoder (VAE)
- Generative Adversarial Network (GAN)

## Repository Breakdown
The following files are included in this project:
1. main.py: This serves as the main controller for the network training. It includes loading the dataset, setting the hyper-parameters, and organizing the training loop over the network
2. networks.py: This file includes the detailed architecture of each network we hope to train
3. helperFxns.py: This file includes several classes that help us organize our program. The NetworkFactory and OptimizerFactory provide access to various network architectures and optimizers. The RunBuilder and RunManager allow us to track the progress of our training in both a dataframe and tensorboard. It organizes the steps that happen before/after each epoch/run to compare hyper-paramter tuning results.

## TODO
This repo is still in development. We hope to include the following functions moving forward:
- Multiprocessing with >1 GPU
- Tensorboard Profiler for memory/time management
- Custom Dataloaders / datasets
- Decaying Learning Rate
- Save best model weights during training

## Refernces
[1] https://deeplizard.com/ 
