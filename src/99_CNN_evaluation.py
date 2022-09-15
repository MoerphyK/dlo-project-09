from cProfile import label
from torch.utils.data import DataLoader
import math
import time
import torch
from torchvision import datasets, transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchinfo import summary
import winsound

#################
### Settings ####
#################

### Try Runtime on CUDA ###
# torch.cuda.is_available = lambda : False # If GPU is found but CPU should be used instead
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

## PATHs
model_path = "cnn_03_baseline_lr_0001.pth"
data_path = "../assets/baseline/"

# General
label_count = 4
batch_size = 1
do_rate = 0
treshhold = 0.90

#############
### Tools ###
#############

def label_to_string(label):
    if label == 0:
        return "rock"
    elif label == 1:
        return "paper"
    elif label == 2:
        return "scissors"
    elif label == 3:
        return "undefined"

def truncate(f, n): 
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

####################
### Data Loading ###
####################

def load_data():
    # Transforming Input into tensors for CNN usage
    transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Resize([128, 128]),
         transforms.Grayscale()
        ])

    dataset = datasets.ImageFolder(data_path, transform=transform)
    global dataset_index
    dataset_index = dataset.class_to_idx

    return dataset

###########
### CNN ###
###########

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Conv2d(ChannelSize,Outputsize,Kernelsize5:5x5)
        # Outputsize must be equal to input size of next layer

        ## 61 = floor( (128 - 8 + 2 * 0)/2 + 1)
        ## 61 = Abgerundet: (ImageSize - KernelSize + Stride * Padding)/Stride? + 1)
        self.conv1 = nn.Conv2d(3, 32, 5)

        # MaxPool2d(Kernelsize,Stride)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)

        # Linear(Inputsize:Outputsize of laster layer * Kernel Size,Outputsize)
        self.fc1 = nn.Linear(64 * 29 * 29, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, label_count)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(do_rate)

    def forward(self, x):
        # calling conv layer with relu optimization function

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape) # Just to check the dimensions
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No Sigmoid needed -> its included in the nn.CrossEntropyLoss()

        return x

##################
### Evaluation ###
##################

def check_probabilities(predicted):
    current_max = 0
    return_label = -1

    # Find highest probabilit
    for i in range(len(predicted[0])):
        # print(f' Probability {label_to_string(i)}: {truncate(predicted[0][i].item(),5)}')
        if predicted[0][i].item() > current_max:
            current_max = predicted[0][i].item()
            return_label = i
    
    # If treshhold is not reached predict undefined class
    if current_max < treshhold:
        return_label = 3 # Label value for undefined

    # print(f'Guessed: {label_to_string(return_label)}')
    return return_label

if __name__ == "__main__":
    # Load data
    dataset = load_data()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f'Amount of data is {len(dataset)}')

    # Load model
    model = ConvNet()
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        summary(model, input_size=(batch_size, 3, 128, 128))
    # evaluate_model(model, device, data_loader)


    model.eval()

    total_acc = 0
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(4)]
    n_class_samples = [0 for i in range(4)]
    counter = 0
    with torch.no_grad(): 
        for images, labels in data_loader: 
            counter += 1
            if counter % (round(len(dataset)/10)+1) == 0:
                print(f'Progress: {counter}/{len(dataset)}')
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            predicted = torch.softmax(outputs,dim=1)
            predicted = check_probabilities(predicted)

            n_samples = n_samples + labels.size(0)
            n_correct = n_correct + (predicted == labels).sum().item()
            # print(f'Label: {label_to_string(labels)}')
            # print(f'Correct {n_correct}/{n_samples}')

            n_samples += 1
            n_class_samples[labels] += 1
            if (predicted == labels):
                n_correct += 1
                n_class_correct[labels] += 1

    # Accuracy Printing:
    total_acc = 100.0 * n_correct / n_samples
    print(f'Total accuracy {total_acc}%; {len(dataset)} images.')
    for i_label in range(4):
        if n_class_samples[i_label] == 0:
            acc = 0
        else:
            acc = 100.0 * n_class_correct[i_label] / n_class_samples[i_label]
        print(f'Accuracy of {label_to_string(i_label)}: {acc}%; {n_class_correct[i_label]}/{n_class_samples[i_label]} correct.')
