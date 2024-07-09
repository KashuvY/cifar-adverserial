import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(img):
    # Define the mean and std used for normalization
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # Ensure the input is on CPU
    img = img.cpu()

    # Denormalize the image
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)

    # Clip values to [0, 1] range
    img = torch.clamp(img, 0, 1)

    # Convert to numpy array and transpose
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()