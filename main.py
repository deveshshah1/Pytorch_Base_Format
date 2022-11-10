"""
Author: Devesh Shah
Project Title: Pytorch Base Format
Branch: transfer_learning

This file represents the basic training function to use for a pytorch model.
In particular, this file loads in a pre-defined network and performs hyper-parameter tuning over the network to
find the best model.

This file is currently set up to load and train for CIFAR10. It can easily be configured for other datasets
by changing the dataloaders

The RunManager format seen here is adapted from https://deeplizard.com/
This format allows us to easily track hyper-parameters as the models are trained
"""


from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from tqdm import tqdm
from methods_and_networks.runManager import *


def main():
    # Set initial set up parameters
    use_tensorboard = False  # results will still be saved in excel if tb is not used
    score_by = {'Accuracy': False, 'Loss': True}  # maximize accuracy (True/False) or minimize Loss (False/True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Read in the datasets and generate class labels list using temporary transforms function
    tforms = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root='./data/Cifar10', train=True, transform=tforms, download=True)
    class_labels = list(train_dataset.class_to_idx.keys())

    # Define all hyper-parameters you wish to study
    # To optimize GPU efficiency, you can alter the num_workers and batch size (e.g. smaller networks typically prefer
    # smaller batch sizes and greater num_workers)
    params = OrderedDict(
        train_set_options=['normalized'],
        num_workers=[2],
        shuffle=[True],
        network=['resnet18', 'resnet50', 'resnet101', 'vgg16', 'inceptionV3', 'squeezenet'],
        pretrained=[True],
        finetune_all_layers=[True, False],
        optimizer=['Adam', 'SGD'],
        l2_reg=[0, 0.001],
        lr=[0.001, 0.0001],
        batch_size=[128],
        epochs=[50]
    )

    params = OrderedDict(
        train_set_options=['normalized'],
        num_workers=[2],
        shuffle=[True],
        network=['Network1'],
        optimizer=['Adam'],
        l2_reg=[0],
        lr=[0.001],
        batch_size=[64, 64, 64],
        epochs=[2]
    )

    m = RunManager(device, use_tensorboard, score_by)
    for run in RunBuilder.get_runs(params):
        network, preprocess = NetworkFactory.get_network(run.network, class_labels, run.pretrained, run.finetune_all_layers)
        network = network.to(device)

        # Read in the datasets and split into train/val/test sets
        train_dataset = torchvision.datasets.CIFAR10(root='./data/Cifar10', train=True, transform=preprocess, download=True)
        train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])
        test_set = torchvision.datasets.CIFAR10(root='./data/Cifar10', train=False, transform=preprocess, download=True)
        train_set_options = {'normalized': train_set}
        val_set_options = {'normalized': val_set}

        param_to_update = []
        for param in network.parameters():
            if param.requires_grad:
                param_to_update.append(param)

        loader = torch.utils.data.DataLoader(train_set_options[run.train_set_options], batch_size=run.batch_size, num_workers=run.num_workers, shuffle=run.shuffle)
        val_loader = torch.utils.data.DataLoader(val_set_options[run.train_set_options], batch_size=run.batch_size, num_workers=run.num_workers, shuffle=False)
        optimizer = OptimizerFactory.get_optimizer(run.optimizer, param_to_update, lr=run.lr, weight_decay=run.l2_reg)

        m.begin_run(run, network, loader, val_loader, list(params.keys()), class_labels)
        print(f"RUN PARAMETERS: {run}")
        for epoch in tqdm(range(run.epochs)):
            m.begin_epoch()

            # Training
            network.train()
            for batch in loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)

                preds = network(images)  # pass batch
                if run.network == 'inceptionV3':
                    loss = F.cross_entropy(preds[0], labels) + 0.4*F.cross_entropy(preds[1], labels)  # calculate loss
                    optimizer.zero_grad()  # zero gradient
                    loss.backward()  # calculate gradients
                    optimizer.step()  # update weights

                    m.track_loss(loss, 'train')
                    m.track_num_correct(preds[0], labels, 'train')
                else:
                    loss = F.cross_entropy(preds, labels)  # calculate loss
                    optimizer.zero_grad()  # zero gradient
                    loss.backward()  # calculate gradients
                    optimizer.step()  # update weights

                    m.track_loss(loss, 'train')
                    m.track_num_correct(preds, labels, 'train')

            # Validation
            network.eval()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch[0].to(device)
                    labels = batch[1].to(device)

                    preds = network(images)
                    loss = F.cross_entropy(preds, labels)

                    m.track_loss(loss, 'val')
                    m.track_num_correct(preds, labels, 'val')

            m.end_epoch()
        m.end_run()
    m.save('training_results')


if __name__ == "__main__":
    main()
