# Pytorch_Base_Format
Many pytorch models can be trained in a similar fashion. This repo includes a baseline format that can be used in many different pytorch models. The goal is to create a standard format to begin new pytorch projects. In particular, we modify the RunManager and RunBuilder class structure provided by DeepLizard [1]. This structure allows us to organize the program into simple classes which can be easily interpretted. This repo is a basic scenario of model development and needs to be modified for more complex scenarios. 

The main branch of this repo includes a training for a simple CNN model. Each additional branch represents templates for various alternative models that can be trained. We include capabilities for the following: 
- Convolutional Neural Network (CNN): Main Branch
- Transfer Learning
- AutoEncoder (AE)
- Variational AutoEncoder (VAE)
- Generative Adversarial Network (GAN)

## Repository Breakdown
The following files are included in this project:
1. main.py: This serves as the main controller for the network training. It includes loading the dataset, setting the hyper-parameters, and organizing the training loop over the network
2. methods_and_networks/networks.py: This file includes the detailed architecture of each network we hope to train
3. methods_and_networks/runManager.py: This file includes several classes that help us organize our program. The NetworkFactory and OptimizerFactory provide access to various network architectures and optimizers. The RunBuilder and RunManager allow us to track the progress of our training in both a dataframe and tensorboard. It organizes the steps that happen before/after each epoch/run to compare hyper-paramter tuning results.
4. methods_and_networks/helperFxns.py: This file includes several helper functions that are implemented and used by the runManager. They are kept in this file to keep the runManager code readable and easily adaptable. 
5. methods_and_networks/helperFxnsDictionary.py: This file is not used by the actual program. Instead it serves as a dictionary by holding potential implemenations of many different methods that may be useful (e.g. feature extraction using hooks, ...). These functions can be taken from here and moved to helperFxns.py and with minimal changes be implemented in the training loop as desired. 

## TODO
This repo is still in development. We hope to include the following functions moving forward:
- Multiprocessing with >1 GPU using torch.nn.parallel.DistributedDataParallel
- Save best model weights during training
- Predict.py file

## Refernces
[1] https://deeplizard.com/ 
