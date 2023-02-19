"""
Author: Devesh Shah
Project Title: Pytorch Base Format

This file represents the basic function we can use to run a trained pytorch model.
In particular, this file loads in a trained network and goes reports the output of the model on a couple
test images.
"""


import torchvision.transforms as transforms
from methods_and_networks.runManager import *
import matplotlib.pyplot as plt


def main(path_to_trained_network='trained_network.pth', n=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Use the same transforms you used for your validation set during trianing. This includes things such as resize
    # to ensure the model is reading in the same relative inputs.
    t_forms = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, ), (0.5, ))])

    # Define the same network architecture to match the saved weights we have
    network = NetworkFactory.get_network('Network1')
    network.load_state_dict(torch.load(path_to_trained_network))
    network = network.to(device)
    network.eval()

    # Read in the datasets and use the test set for this
    test_set = torchvision.datasets.CIFAR10(root='./data/Cifar10', train=False, transform=t_forms, download=True)
    class_labels = list(test_set.class_to_idx.keys())

    to_visualize, _ = torch.utils.data.random_split(test_set, [n, (len(test_set)-n)])
    loader = torch.utils.data.DataLoader(to_visualize, batch_size=1, num_workers=2, shuffle=False)

    for i, batch in enumerate(loader):
        print(f"Loading Image #{i+1}:")
        images = batch[0].to(device)
        labels = batch[1].to(device)

        output = network(images)
        output = torch.nn.functional.softmax(output, dim=1)
        values, indicies = torch.max(output, 1)

        print(f'Predicted: {int(values[0]*100)}% {class_labels[indicies[0]]}')
        print(f'Actual: {class_labels[labels[0]]}')

        img = images[0].permute(1, 2, 0)

        plt.figure()
        plt.imshow(img.cpu())
        plt.title(f'Predicted: {int(values[0]*100)}% {class_labels[indicies[0]]}\nActual: {class_labels[labels[0]]}')
        plt.show()


if __name__ == "__main__":
    main()
