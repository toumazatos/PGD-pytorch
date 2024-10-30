import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.utils
from torchvision import models, datasets as dsets, transforms
import matplotlib.pyplot as plt
import numpy as np
import json

# Set device to CPU
use_cuda = False
device = torch.device("cpu")

# Prepare ImageNet Data
class_idx = json.load(open("./data/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]
])

def image_folder_custom_label(root, transform, custom_label):
    old_data = dsets.ImageFolder(root=root, transform=transform)
    label2idx = {item: i for i, item in enumerate(idx2label)}
    new_data = dsets.ImageFolder(root=root, transform=transform,
                                 target_transform=lambda x: custom_label.index(idx2label[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx
    return new_data

# Load data
normal_data = image_folder_custom_label(root='./data/imagenet', transform=transform, custom_label=idx2label)
normal_loader = Data.DataLoader(normal_data, batch_size=1, shuffle=False)

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

    # Print results for each image
    print(f"Step losses: {step_losses}")
    print(f"Step predictions: {step_predictions}")

    return images

# Evaluate on adversarial examples
correct = 0
total = 0
image_cap = 50  # Set your desired cap here

for i, (images, labels) in enumerate(normal_loader):
    if i >= image_cap:
        break

    images = pgd_attack(model, images, labels)
    outputs = model(images)
    _, pre = torch.max(outputs.data, 1)
    total += 1
    correct += (pre == labels).sum()

    # Show adversarial image
    imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), [normal_data.classes[i] for i in pre])

print('Accuracy of adversarial test images: %f %%' % (100 * float(correct) / total))
