import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
from model import vgg

root_path = os.path.join(os.getcwd(), 'dataset')
data_path = os.path.join(root_path, 'data_set', 'flower_data', 'flower_photos')
# print(data_path)

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using {} device for train.".format(device))

batch_size = 32
num_worker = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

train_set = datasets.ImageFolder(root=os.path.join(data_path, 'train'),
                                 transform=data_transform['train'],
                                 )
train_loader = DataLoader(train_set, shuffle=True,
                          batch_size=batch_size, num_workers=0)

val_set = datasets.ImageFolder(root=os.path.join(data_path, 'val'),
                               transform=data_transform['val'])
val_loader = DataLoader(val_set, shuffle=False,
                        num_workers=0)

model_name = 'vgg16'
net = vgg(model_name=model_name, num_classes=5, init_weights=True)
net.to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

epochs = 30
for epoch in range(epochs):
    running_loss = 0.0
    best_acc = 0.0
    save_path = './weight/AlexNet.pth'

    net.train()

    train_bar = tqdm(train_loader, file=sys.stdout)
    for steps, (images, labels) in enumerate(train_bar):
        optimizer.zero_grad()
        predictions = net(images.to(device))
        loss = loss_function(predictions, labels.to(device))
        loss.backward()

        running_loss += loss.item()

        train_bar.desc = "train epoch: {}/{}, loss: {}".format(epoch + 1, epochs, loss)

    net.eval()
    with torch.no_grad():
        for (inputs, targets) in val_loader:
            outputs = net(inputs.to(device))
            outputs = torch.max(outputs, dim=1)[1]
            acc = torch.eq(targets.to(device), outputs).sum().item()

    val_accuracy = acc / len(val_loader)
    print("[epoch:{}] train_loss: {}, val_accuracy: {}".format(epoch, running_loss / steps, val_accuracy))

    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(net.state_dict(), save_path)

print("Fnishing training")






















