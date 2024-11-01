import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.utils
from torchvision import models, datasets as dsets, transforms
import matplotlib.pyplot as plt
import numpy as np
import json
from torchvision.datasets import CIFAR10


# Set device to CPU
use_cuda = False
device = torch.device("cpu")

# Prepare ImageNet Data

transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to match the input size for Inception v3
    transforms.ToTensor(),
])

normal_data = CIFAR10(root='./data', train=False, transform=transform, download=True)
normal_loader = Data.DataLoader(normal_data, batch_size=1, shuffle=False)
print(f"Total images available: {len(normal_loader.dataset)}")

# Define a function to display images
def imshow(img, title):
    npimg = img.numpy()
    plt.figure(figsize=(5, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

# Load pre-trained Inception v3 model
model = models.inception_v3(weights="IMAGENET1K_V1").to(device)
model.eval()

# PGD Attack Function with loss and prediction tracking
def pgd_attack(model, images, labels, eps=0.3, alpha=2/255, iters=40):
    images = images.to(device)
    labels = labels.to(device)
    loss_fn = nn.CrossEntropyLoss()

    ori_images = images.data

    # Arrays to store loss and predictions at each step
    step_losses = []
    step_predictions = []

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        # Calculate loss and gradients
        model.zero_grad()
        loss = loss_fn(outputs, labels).to(device)
        loss.backward()

        # Record loss and prediction for each step
        step_losses.append(loss.item())
        _, pred = torch.max(outputs.data, 1)
        step_predictions.append(pred.item())

        # Apply perturbation
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    # Return adversarial images, losses, and predictions for each step
    return images, step_losses, step_predictions

# Evaluate on adversarial examples
correct = 0
total = 0
image_cap = 5  # Set your desired cap here
all_step_losses = []
all_step_predictions = []

for i, (images, labels) in enumerate(normal_loader):
    if i >= image_cap:
        break

    # Perform PGD attack and record losses and predictions at each step
    adv_images, step_losses, step_predictions = pgd_attack(model, images, labels)
    all_step_losses.append(step_losses)
    all_step_predictions.append(step_predictions)

    outputs = model(adv_images)
    _, pre = torch.max(outputs.data, 1)
    total += 1
    correct += (pre == labels).sum()

    # Show adversarial image
    print(f"Predicted indices for image {i + 1}: {pre.cpu().numpy()}")
    imshow(torchvision.utils.make_grid(adv_images.cpu().data, normalize=True), title="Adversarial Image")

    # Print per-image losses and predictions
    print(f"Image {i + 1}: Step losses = {step_losses}")
    print(f"Image {i + 1}: Step predictions = {step_predictions}")

print('Accuracy of adversarial test images: %f %%' % (100 * float(correct) / total))

# You now have all_step_losses and all_step_predictions, each containing 50 arrays
# of 40 elements (if iters=40 and image_cap=50) for further analysis or plotting.
