import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from model import AlexNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using {} device.".format(device))

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

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
image_path = os.path.join(data_root, 'data_set', 'flower_data')
assert os.path.exists(image_path), "{} path is not exists.".format(image_path)

train_set = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                 transform=data_transform["train"])
train_num = len(train_set)

flower_list = train_set.class_to_idx
# print(flower_list)
cla_dict = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(cla_dict, indent=4)

with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
num_worker = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=num_worker)
val_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                                   transform=transforms["val"]
                                   )
val_num = len(val_dataset)
val_loader = DataLoader(val_dataset, shuffle=False,
                        batch_size=4, num_workers=num_worker
                        )

print("using {} images for train, {} images for validation.".format(train_num, val_num))

# 数据集、网络、损失函数 优化器、训练、测试

net = AlexNet(num_worker=5, init_weights=True)
net.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002)

epochs = 10
save_path = './AlexNet.pth'
best_acc = 0.0
train_steps = len(train_loader)
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, (images, labels) in enumerate(train_bar):
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(labels.to(device), outputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

print('Finished Training')
