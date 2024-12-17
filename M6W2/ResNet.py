import torch
import torch.nn as nn
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class WeatherDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.img_paths = X
        self.labels = y
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


def transform(img, img_size=(224, 224)):
    img = img.resize(img_size)
    img = np.array(img)
    img = torch.tensor(img).permute(2, 0, 1).float()
    normalized_img = img / 255.0
    return normalized_img


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = x.clone()
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        shortcut = self.downsample(shortcut)
        x += shortcut
        x = nn.ReLU(inplace=True)(x)
        return x


class ResNet(nn.Module):
    def __init__(self, residual_block, n_blocks_lst, n_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = self.create_layer(
            residual_block, 64, 64, n_blocks_lst[0], stride=1)
        self.conv3 = self.create_layer(
            residual_block, 64, 128, n_blocks_lst[1], stride=2)
        self.conv4 = self.create_layer(
            residual_block, 128, 256, n_blocks_lst[2], stride=2)
        self.conv5 = self.create_layer(
            residual_block, 256, 512, n_blocks_lst[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, n_classes)

    def create_layer(self, residual_block, in_channels, out_channels, n_blocks, stride=1):
        blocks = []
        first_block = residual_block(in_channels, out_channels, stride)
        blocks.append(first_block)
        for idx in range(1, n_blocks):
            block = residual_block(out_channels, out_channels, stride)
            blocks.append(block)

        block_sequential = nn.Sequential(*blocks)
        return block_sequential

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.max_pool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return x


def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0.0
    total = 0
    losses = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    loss = np.mean(losses)
    return loss, accuracy


def fit(model, train_dataloader, val_dataloader, criterion, optimizer, device, epochs):
    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        batch_train_losses = []
        model.train()

        for idx, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_train_losses.append(loss.item())

        train_loss = np.mean(batch_train_losses)
        train_losses.append(train_loss)

        val_loss, val_accuracy = evaluate(
            model, val_dataloader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f'EPOCH {epoch + 1}:\tTrain loss : {train_loss:.4f}\tVal loss : {val_loss:.4f}')

    return train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == '__main__':
    seed = 59
    set_seed(seed)

    roor_dir = '/weather-dataset/dataset'
    img_paths = []
    labels = []
    classes = {
        label_idx: class_name
        for label_idx, class_name in enumerate(sorted(
            roor_dir))
    }

    for label_idx, class_name in classes.items():
        class_dir = os.path.join(roor_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img_paths.append(img_path)
            labels.append(label_idx)

    val_size = 0.2
    test_size = 0.125
    is_shuffle = True

    X_train, X_val, y_train, y_val = train_test_split(
        img_paths, labels, test_size=val_size, shuffle=is_shuffle
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=test_size, shuffle=is_shuffle
    )
    train_dataset = WeatherDataset(X_train, y_train, transform=transform)
    val_dataset = WeatherDataset(X_val, y_val, transform=transform)
    test_dataset = WeatherDataset(X_test, y_test, transform=transform)

    train_batch_size = 512
    test_batch_size = 8

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False
    )

    n_classes = len(list(classes.keys()))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ResNet(ResidualBlock, [2, 2, 2, 2], n_classes).to(device)

    lr = 1e-2
    epochs = 25
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_losses, val_losses, train_accuracies, val_accuracies = fit(
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        device,
        epochs
    )

    val_loss, val_acc = evaluate(model, val_dataloader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print('Evaluation on val / test dataset')
    print('Val accuracy : ', val_acc)
    print('Test accuracy : ', test_acc)

    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.legend()
    plt.show()
