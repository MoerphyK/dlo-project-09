### IMPORT Scripts ###

# import data_loader as dl

### IMPORTS LIB ###

from matplotlib import image
import numpy as np
import pandas as pd
import math

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#from torchsummary import summary
from torchinfo import summary

### Try Runtime on CUDA ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print( f"device: {device}" )

### Load Data ### 

# train_dataset, test_datset, train_loader, test_loader = dl.import_data()

#Hyperparameters
num_epochs = 5
batch_size = 5
learning_rate = 0.001

# Transforming Input into tensors for CNN usage
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
)

def load_train_data():
    # type = train or test
    transform = transforms.Compose([transforms.Resize([128,128]), transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    # ggf. transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
    dataset = datasets.ImageFolder(f'../assets/train/',transform=transform)
    global dataset_index
    dataset_index = dataset.class_to_idx
    return dataset

def load_test_data():
    # type = train or test
    transform = transforms.Compose([transforms.Resize([128,128]),transforms.ToTensor()])
    dataset = datasets.ImageFolder(f'../assets/test/',transform=transform)
    return dataset

# use imshow for printing images for later use:
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

## Doesn't work with tensor, only a raw dataset
def show_img(d):
    #Print (dataset [0] [1]: hier d[1]) # the first dimension is the number of images, the second dimension is 1, and label is returned
    #Print (dataset [0] [0]: hier d[0]) # is 0 and returns picture data
    plt.imshow(d[0].permute(1, 2, 0),interpolation='nearest')
    plt.title([k for k, v in dataset_index.items() if v == d[1]][0])
    plt.axis('off')
    plt.show()

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

def label_to_string(label):
    if label == 0:
        return "rock"
    elif label == 1:
        return "paper"
    elif label == 2:
        return "scissors"
    elif label == 3:
        return "undefined"

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        #Conv2d(ChannelSize,Outputsize,Kernelsize5:5x5)
        #Outputsize must be equal to input size of next layer

        ## 61 = floor( (128 - 8 + 2 * 0)/2 + 1)
        ## 61 = Abgerundet: (ImageSize - KernelSize + Stride * Padding)/Stride? + 1)
        self.conv1 = nn.Conv2d(3, 32, 5)

        #MaxPool2d(Kernelsize,Stride)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32,64,5)

        #Linear(Inputsize:Outputsize of laster layer * Kernel Size,Outputsize)
        self.fc1 = nn.Linear(64*29*29, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 4)

         # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #calling conv layer with relu optimization function

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape) # Just to check the dimensions
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x) # No Sigmoid needed -> its included in the nn.CrossEntropyLoss()

        return x


if __name__ == "__main__":
    ### Try Runtime on CUDA ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print( f"device: {device}" )

    ### Load data ### 
    train_dataset = load_train_data()
    test_dataset = load_test_data()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Test for dataloading
    # images, labels = next(iter(train_loader))
    # show_img(test_dataset[0])
    # show_img(test_dataset[4])
    # show_img(test_dataset[10])
    # show_img(test_dataset[15])
    # print(f"Types: Image: {type(images[0])}; Label: {type(labels[0])}")
    # print(f"Shape: Image: {images[0].shape}; Label: {labels[0].shape}") 
    # print(f"Values: Image: {images[0]}; Label: {label_to_string(labels[0])}") 
    # exit(0)

    ### Setup for ConvModule ###

    # classes = ('rock','paper','scissors','miscellaneous')


    model = ConvNet().to(device)
    x = torch.randn(1, 3, 128, 128)

    #summary(model,(3,128,128))
    summary(model, input_size=(batch_size, 3, 128, 128))

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

            if (i+1) % math.floor(n_total_steps/4) == 0 or (i+1) % n_total_steps == 0 or (i+1) % 25 == 0:
                print(f'Epoch: [{epoch+1}/{num_epochs}],Step: [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        print(f'Epoch: [{epoch+1}/{num_epochs} finished.]')

    print('Finished Training')

    ### Setup path to save model ###

    PATH = './cnn.pth'
    torch.save(model.state_dict(), PATH)

    ### Evaluating the Model ###

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(4)]
        n_class_samples = [0 for i in range(4)]

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            n_samples = n_samples + labels.size(0)
            n_correct = n_correct + (predicted == labels).sum().item()

            for i in range(len(images)): #ehem. batch_size
                label = labels[i]
                pred = predicted[i]

                if (label == pred):
                    n_class_correct[label] = n_class_correct[label] + 1
                
                n_class_samples[label] = n_class_samples[label] + 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc}%')

        for i in range(4):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {label_to_string(i)}: {acc}%')
