import os

import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import copy
from torchvision import transforms, models
from skimage import io


from tqdm import tqdm
from tkinter import Image
from torch.utils.data import Dataset
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


from sklearn.model_selection import train_test_split


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



negative_paths = os.listdir('D:\ДИПЛОМ\Проект_1\Снимки экрана\Без телефона')
positive_paths = os.listdir('D:\ДИПЛОМ\Проект_1\Снимки экрана\С телефоном')
negative_labels = [0] * len(negative_paths)
positive_labels = [1] * len(positive_paths)

negative_paths.extend(positive_paths)
negative_labels.extend(positive_labels)


paths = pd.DataFrame(negative_paths, columns=['paths'])
labels = pd.DataFrame(negative_labels, columns=['labels'])

df = pd.concat([paths, labels], axis=1)
df = df.sample(frac=1).reset_index(drop=True)
print(df.head())

df.to_csv('awesome_csv.csv', index=False)

# X = df.iloc[:, 0: -1]
# y = df.iloc[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)


class PathologyPlantsDataset(Dataset):

    def __init__(self, data_frame, root_dir, transform=None):
        self.data_frame = pd.read_csv(data_frame)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # Return the length of the dataset
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        # image = io.imread(img_name)
        label = self.data_frame.iloc[idx, -1]
        # label = np.array([label])
        # label = label.astype('float').reshape(-1, 1)
        # sample = {'image': image, 'labels': label}
        if self.transform:
            image = self.transform(image)

        return (image, label)


# INSTANTIATE THE OBJECT
pathology_train = PathologyPlantsDataset(
    data_frame='D:\ДИПЛОМ\Проект_1\\awesome_csv.csv',
    root_dir='D:\ДИПЛОМ\Проект_1\\all',
    transform=transforms.Compose({
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    })
)


batch_size = 8  # форматирую (привожу к тензорам) информацию для загрузки в нейросеть
train_dataloader = torch.utils.data.DataLoader(
    pathology_train, batch_size=batch_size, shuffle=True, num_workers=batch_size)
val_dataloader = torch.utils.data.DataLoader(
    pathology_train, batch_size=batch_size, shuffle=False, num_workers=batch_size)

X_batch, y_batch = next(iter(train_dataloader))
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
plt.imshow(X_batch[0].permute(1, 2, 0).numpy() * std + mean);


def show_input(input_tensor, title=''):
    image = input_tensor.permute(1, 2, 0).numpy()
    image = std * image + mean
    plt.imshow(image.clip(0, 1))
    plt.title(title)
    plt.show()
    plt.pause(0.001)


class_names = ['phones', 'no_phones']


X_batch, y_batch = next(iter(train_dataloader))  # подмножество обучающей выборки

for x_item, y_item in zip(X_batch, y_batch):
    show_input(x_item, title=class_names[y_item])


def train_model(model, loss, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                dataloader = val_dataloader
                model.eval()  # Set model to evaluate mode

            running_loss = 0.
            running_acc = 0.

            # Iterate over data.
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                # statistics
                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

    return model


model = models.resnet50(pretrained=True)

# Disable grad for all conv layers
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_model(model, loss, optimizer, scheduler, num_epochs=100);



