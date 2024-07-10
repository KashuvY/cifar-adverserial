import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from models import Net, ResNet
from plot_images import imshow

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10 dataset.")
    parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    batch_size = 32

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    PATH = './cifar_net.pth'

    
    # Get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Show images
    imshow(torchvision.utils.make_grid(tensor=images, nrow=8))  # make_grid(images) treats it as one large image with compressed subimages
                                                    # valid arguments: tensor, nrow, padding, normalize, range, scale_each, pad_value

    # Print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    
    del dataiter

    net = ResNet()

    # If you want to fine-tune only the last layer, you can freeze the other layers:
    for param in net.resnet.parameters():
        param.requires_grad = False
    for param in net.resnet.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    # Don't forget to update your optimizer to only update the trainable parameters
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001, momentum=0.9)

    for epoch in range(args.n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training')

    torch.save(net.state_dict(), PATH)



if __name__ == '__main__':
    main()
