"""
Author: Devesh Shah
Project Title: Pytorch Base Format

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
from helperFxns import *


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Define transforms to use when reading in image dataset
    # If you would like to apply random augmentations (e.g. horizontal/verical flip, random crop, ...) then you
    # must make two different t_forms. One for the training dataset, and one for the test set (without augmentations)
    t_forms = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, ), (0.5, ))])

    # Read in the datasets and split into train/val/test sets
    train_dataset = torchvision.datasets.CIFAR10(root='./data/Cifar10', train=True, transform=t_forms, download=True)
    class_labels = list(train_dataset.class_to_idx.keys())
    train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])
    test_set = torchvision.datasets.CIFAR10(root='./data/Cifar10', train=False, transform=t_forms, download=True)
    train_set_options = {'normalized': train_set}
    val_set_options = {'normalized': val_set}

    # Define all hyper-parameters you wish to study
    params = OrderedDict(
        train_set_options=['normalized'],
        num_workers=[2],
        shuffle=[True],
        network=['Network1'],
        optimizer=['Adam'],
        l2_reg=[0],
        lr=[0.001],
        batch_size=[64],
        epochs=[1]
    )

    m = RunManager(device)
    for run in RunBuilder.get_runs(params):
        network = NetworkFactory.get_network(run.network)
        network = network.to(device)

        loader = torch.utils.data.DataLoader(train_set_options[run.train_set_options], batch_size=run.batch_size, num_workers=run.num_workers, shuffle=run.shuffle)
        val_loader = torch.utils.data.DataLoader(val_set_options[run.train_set_options], batch_size=run.batch_size, num_workers=run.num_workers, shuffle=False)
        optimizer = OptimizerFactory.get_optimizer(run.optimizer, network.parameters(), lr=run.lr, weight_decay=run.l2_reg)

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
