import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from cnn import ResNet  # Assuming you've saved the ResNet class in a file named resnet.py
from plot_images import imshow

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test a ResNet on CIFAR-10 dataset.")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize(224),  # Resize to 224x224 for ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    batch_size = 4

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    PATH = './cifar_resnet.pth'

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(tensor=images))
    print('Actual: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    net = ResNet()
    net.load_state_dict(torch.load(PATH))
    
    # Set the model to evaluation mode
    net.eval()

    # Enable gradient computation for the input
    images.requires_grad = True

    # Forward pass
    outputs = net(images)

    # Get probabilities
    probabilities = nn.functional.softmax(outputs, dim=1)

    # Get predicted classes
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))
    
    # Print probabilities for each prediction
    for i in range(batch_size):
        print(f"Image {i+1} probabilities:")
        for j, prob in enumerate(probabilities[i]):
            print(f"  {classes[j]}: {prob.item():.4f}")

    # Compute gradients for creating an adversarial example
    target_class = 5  # Let's say we want to misclassify everything as 'dog'
    loss = nn.functional.cross_entropy(outputs, torch.full_like(labels, target_class))
    loss.backward()

    # Create adversarial example using FGSM
    epsilon = 0.01
    adversarial_images = images + epsilon * images.grad.sign()

    # Clamp to ensure valid pixel range
    adversarial_images = torch.clamp(adversarial_images, 0, 1)

    # Test the adversarial example
    with torch.no_grad():
        adv_outputs = net(adversarial_images)
        adv_probabilities = nn.functional.softmax(adv_outputs, dim=1)
        _, adv_predicted = torch.max(adv_outputs, 1)

    print("\nAdversarial Example Predictions:")
    imshow(torchvision.utils.make_grid(tensor=adversarial_images))
    for i in range(batch_size):
        print(f"Image {i+1} probabilities:")
        for j, prob in enumerate(adv_probabilities[i]):
            print(f"  {classes[j]}: {prob.item():.4f}")
    print(' '.join(f'{classes[adv_predicted[j]]:5s}' for j in range(batch_size)))

    # Compute overall accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'\nAccuracy of the network on the 10000 test images: {100 * correct // total} %')

if __name__ == '__main__':
    main()