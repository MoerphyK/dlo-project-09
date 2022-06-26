### IMPORT Scripts ###

import data_loader as dl

### IMPORTS LIB ###

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

### Try Runtime on CUDA ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print( f"device: {device}" )

### Load Data ### 

train_dataset, test_datset, train_loader, test_loader = dl.import_data()

#Hyperparameters
num_epochs = 5
batch_size = 5
learning_rate = 0.001

# Transforming Input into tensors for CNN usage
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
)

### Setup for ConvModule ###

classes = ('rock','paper','scissors','miscellaneous')

# use imshow for printing images for later use:
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        #Conv2d(ChannelSize,Outputsize,Kernelsize5:5x5)
        #Outputsize must be equal to input size of next layer

        self.conv1 = nn.Conv2d(3, 6, 5)

        #MaxPool2d(Kernelsize,Stride)

        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6,16,5)

        #Linear(Inputsize:Outputsize of laster layer * Kernel Size,Outputsize)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):

        #calling conv layer with relu optimization function

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # No Sigmoid needed -> its included in the nn.CrossEntropyLoss()
        
        x = self.fc3(x)
    
        return x

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'Epoch: [{epoch+1}/{num_epochs}],Step: [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')

### Setup path to save model ###

# PATH = './cnn.pth'
# torch.save(model.state_dict(), PATH)

### Evaluating the Model ###

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        n_samples = n_samples + labels.size(0)
        n_correct = n_correct + (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]

            if (label == pred):
                n_class_correct[label] = n_class_correct[label] + 1
            
            n_class_samples[label] = n_class_samples[label] + 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc}%')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc}%')