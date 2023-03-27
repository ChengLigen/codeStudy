import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import sys
from model import GoogLeNet



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using {} deevice for train.".format(device))

data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}


root_path = os.getcwd()
image_path = os.path.join(root_path, 'flower_photos')

batch_size = 32
num_worker = min(os.cpu_count(), batch_size if batch_size > 1 else 0, 8)

train_set = datasets.ImageFolder(os.path.join(image_path, 'train'),
                                 transform=data_transform['train'])
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=num_worker)

val_set = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                               transform=data_transform['val'])
val_loader = DataLoader(val_set, shuffle=False, num_workers=num_worker)

net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)




























































