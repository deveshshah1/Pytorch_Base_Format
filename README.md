# Pytorch_Base_Format
Many pytorch models can be trained in a similar fashion. This repo includes a baseline format that can be used in many different pytorch models. The goal is to create a standard format to begin new pytorch projects. In particular, we modify the RunManager and RunBuilder class structure provided by DeepLizard [1]. This structure allows us to organize the program into simple classes which can be easily interpretted. This repo is a basic scenario of model development and needs to be modified for more complex scenarios. 

The main branch of this repo includes a training for a simple CNN model. Each additional branch represents templates for various alternative models that can be trained. We include capabilities for the following: 
- Convoluation Neural Network (CNN)
- AutoEncoder (AE)
- Variational AutoEncoder (VAE)
- Generative Adversarial Network (GAN)

## TODO
This repo is still in development. We hope to include the following functions moving forward:
- Multiprocessing with >1 GPU
- Transfer Learning

## Refernces
[1] https://deeplizard.com/ 
