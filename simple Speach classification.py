from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


class MyDataset (Dataset):
    def __init__(self, path=None, csv_file=None):
        self.path = path
        self.csv_file = csv_file
        self.data = pd.read_csv(self.csv_file)
        self.data['target'] = LabelEncoder().fit_transform(self.data['target'])
        self.n_sambles = len([f for f in os.listdir(
            self.path)if os.path.isfile(os.path.join(self.path, f))])

    def __len__(self):
        return self.n_sambles

    def __getitem__(self, index):
        tar = self.data['target'].iloc[index]
        wav_file = self.data['file_name'].iloc[index]
        waveform, sample_rate = torchaudio.load(
            os.path.join(self.path, wav_file))
        mfcc_spectrogram = torchaudio.transforms.MFCC(
            sample_rate=sample_rate)(waveform)
        return mfcc_spectrogram, tar


train_data = MyDataset(
    path='/mnt/23ce3591-a7a9-4853-8164-8609c66f367c/task1/data/train',
    csv_file='/mnt/23ce3591-a7a9-4853-8164-8609c66f367c/task1/train.csv')
test_data = MyDataset(
    path='/mnt/23ce3591-a7a9-4853-8164-8609c66f367c/task1/data/test',
    csv_file='/mnt/23ce3591-a7a9-4853-8164-8609c66f367c/task1/test.csv')

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=8,
    shuffle=True,
    num_workers=2)

train_loader= torch.utils.data.DataLoader(
    train_data,
    batch_size=8,
    shuffle=True,
    num_workers=2)


class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(43456, 50)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return F.log_softmax(x, dim=1)


model = CNNet()

cost = torch.nn.CrossEntropyLoss()

# used to create optimal parameters
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create the training function


def train(dataloader, model, loss, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, Y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


# Create the validation/test function

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            pred = model(X)
            print(pred)
            test_loss += cost(pred, Y).item()
            correct += (pred.argmax(1) == Y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size

    print(
        f'\nTest Error:\nacc: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}\n')


print(model)

epochs = 3

for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    train(train_loader, model, cost, optimizer)
    test(test_loader, model)
print('Done!')
