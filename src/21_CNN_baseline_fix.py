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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
label_count = 4

## PATHs
MODEL_PATH = "cnn_21_baseline_fix.pth"
DATA_PATH = "../assets/baseline/"
PLOT_PATH = "../docs/test_results/21_baseline_fix_plot.png"
TXT_PATH = "../docs/test_results/21_baseline_fix.txt"

# Hyperparameters
num_epochs = 50
batch_size = 32
learning_rate = 0.0001
do_rate = 0.6

# Early stopping
last_loss = 100
patience = 4
trigger_times = 0

# Temporary variables
best_acc = {}
best_acc['total'] = 0
best_acc['rock'] = 0
best_acc['scissor'] = 0
best_acc['paper'] = 0
best_acc['undefined'] = 0

current_acc = {}
current_acc['total'] = 0
current_acc['rock'] = 0
current_acc['scissor'] = 0
current_acc['paper'] = 0
current_acc['undefined'] = 0

#############
### Tools ###
#############

## Doesn't work with tensor, only a raw dataset
def show_img(d):
    # Print (dataset [0] [1]: hier d[1]) # the first dimension is the number of images, the second dimension is 1, and label is returned
    # Print (dataset [0] [0]: hier d[0]) # is 0 and returns picture data
    plt.imshow(d[0].permute(1, 2, 0), interpolation='nearest')
    plt.title([k for k, v in dataset_index.items() if v == d[1]][0])
    plt.axis('off')
    plt.show()


def label_to_string(label):
    if label == 0:
        return "rock"
    elif label == 1:
        return "paper"
    elif label == 2:
        return "scissors"
    elif label == 3:
        return "undefined"

####################
### Data Loading ###
####################

def load_data():
    # Transforming Input into tensors for CNN usage
    transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Resize([128, 128]),
         transforms.Grayscale()])

    dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
    global dataset_index
    dataset_index = dataset.class_to_idx

    test_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - test_size
    print(f"Test Size: {test_size}; Train Size: {train_size}; Total in Dataset: {len(dataset)}")
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                       generator=torch.Generator().manual_seed(42))
    return train_set, val_set

##################
### Evaluation ###
##################

def evaluate_model(model, device, test_loader, loss_function, plt_lists):
    '''
    model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time.
    For example, Dropouts Layers, BatchNorm Layers etc.
    You need to turn off them during model evaluation, and .eval() will do it for you.
    In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
    '''
    loss_total = 0
    model.eval()

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(label_count)]
        n_class_samples = [0 for i in range(label_count)]

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = loss_function(outputs, labels)
            loss_total += loss.item()

            _, predicted = torch.max(outputs, 1)
            n_samples = n_samples + labels.size(0)
            n_correct = n_correct + (predicted == labels).sum().item()

            for i in range(len(images)):  # ehem. batch_size
                label = labels[i]
                pred = predicted[i]

                if (label == pred):
                    n_class_correct[label] = n_class_correct[label] + 1

                n_class_samples[label] = n_class_samples[label] + 1

        total_acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {total_acc}%')
        plt_lists['overall_acc'].append(total_acc)
        current_acc['total'] = total_acc

        loss_total_per_items = loss_total / len(test_loader)
        plt_lists['loss_value'].append(loss_total_per_items)

        for i_label in range(label_count):
            if n_class_samples[i_label] == 0:
                acc = 0
            else:
                acc = 100.0 * n_class_correct[i_label] / n_class_samples[i_label]
            if i_label == 0:
                plt_lists['rock_acc'].append(acc)
                current_acc['rock'] = acc
            elif i_label == 1:
                plt_lists['paper_acc'].append(acc)
                current_acc['paper'] = acc
            elif i_label == 2:
                plt_lists['scissor_acc'].append(acc)
                current_acc['scissor'] = acc
            elif i_label == 3:
                plt_lists['undefined_acc'].append(acc)
                current_acc['undefined'] = acc
            print(f'Accuracy of {label_to_string(i_label)}: {acc}%; {n_class_correct[i_label]}/{n_class_samples[i_label]} correct.')

        return loss_total_per_items, plt_lists


def plot_accuracy(plt_lists):
    figure = plt.figure()
    figure.suptitle('DLO - Baseline')

    plt.subplot(3, 1, 1)
    plt.plot(plt_lists['rock_acc'], 'b', label='rock_acc')
    plt.plot(plt_lists['paper_acc'], 'g', label='paper_acc')
    plt.plot(plt_lists['scissor_acc'], 'r', label='scissor_acc')
    plt.plot(plt_lists['undefined_acc'], 'k', label='undefined_acc')
    # plt.xlabel('Epochs')
    plt.ylabel('Accuracy / class')
    plt.legend(loc='upper right')
    plt.grid('y')
    plt.title = 'Accuracy per class'

    plt.subplot(3, 1, 2)
    plt.plot(plt_lists['overall_acc'], 'x-')
    # plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid('y')
    plt.title = 'Overall Accuracy'

    plt.subplot(3, 1, 3)
    plt.plot(plt_lists['loss_value'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid('y')
    plt.title = 'Loss'

    plt.savefig(PLOT_PATH)
    plt.show()

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
        self.conv1 = nn.Conv2d(1, 32, 5)

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

############
### Main ###
############

if __name__ == "__main__":
    ### Load data ###
    train_dataset, test_dataset = load_data()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    n_total_steps = len(train_loader)

    ### Setup for ConvModule ###
    model = ConvNet().to(device)
    # model.load_state_dict(torch.load(MODEL_PATH)) # Enable if the training of the loaded pth model shall continue
    summary(model, input_size=(batch_size, 1, 128, 128))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    plt_lists = {}
    plt_lists['rock_acc'] = []
    plt_lists['paper_acc'] = []
    plt_lists['scissor_acc'] = []
    plt_lists['undefined_acc'] = []
    plt_lists['overall_acc'] = []
    plt_lists['loss_value'] = []

    t_total = time.time()
    for epoch in range(num_epochs):
        t0 = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ### Output Epoch Progess
            if (i + 1) % math.floor(n_total_steps / 10) == 0 or (i + 1) % n_total_steps == 0:
                print(f'Epoch: [{epoch + 1}/{num_epochs}],Step: [{i + 1}/{n_total_steps}], Loss: {loss.item():.8f}')
        print(f'Epoch: [{epoch + 1}/{num_epochs} finished in {int(time.time() - t0)} seconds.]')
        
        ### Evaluating + Early stopping the Model ###
        current_loss, plt_lists = evaluate_model(model, device, test_loader, criterion, plt_lists)
        print(f'The Current Loss: {current_loss}')

        if current_loss > last_loss:
            trigger_times += 1
            print(f'Trigger Times: {trigger_times}')

            if trigger_times >= patience:
                print('Early stopping!')
                break
        else:
            ### Setup path to save model ###
            torch.save(model.state_dict(), MODEL_PATH)
            # Reset Early Stopping Trigger
            print(f'Trigger times back to 0.')
            trigger_times = 0
            best_acc['total'] = current_acc['total']
            best_acc['rock'] = current_acc['rock']
            best_acc['paper'] = current_acc['paper']
            best_acc['scissor'] = current_acc['scissor']
            best_acc['undefined'] = current_acc['undefined']
            best_acc['epoch'] = epoch + 1
            

        last_loss = current_loss

    print(f'### Finished Training in {int(time.time() - t_total)} seconds ###')
    print(f'The total accuracy is {best_acc["total"]}%')
    print(f'The rock accuracy is {best_acc["rock"]}%')
    print(f'The paper accuracy is {best_acc["paper"]}%')
    print(f'The scissor accuracy is {best_acc["scissor"]}%')
    print(f'The undefined accuracy is {best_acc["undefined"]}%')
    print(f'At epoch {best_acc["epoch"]}')

    with open(TXT_PATH, 'w') as convert_file:
            convert_file.write(f'Accuracies in epoch {best_acc["epoch"]}:\n')
            convert_file.write(f'Total: {best_acc["total"]}%\n')
            convert_file.write(f'Rock: {best_acc["rock"]}%\n')
            convert_file.write(f'Paper: {best_acc["paper"]}%\n')
            convert_file.write(f'Scissor: {best_acc["scissor"]}%\n')
            convert_file.write(f'Undefined: {best_acc["undefined"]}%')
            convert_file.write(f'\n\nHyperparemter:')
            convert_file.write(f'Batch Size: {batch_size}\n')
            convert_file.write(f'Learning Rate: {learning_rate}\n')
            convert_file.write(f'Drop Out Rate: {do_rate}\n')
            convert_file.write(f'Early Stopping Patience: {patience}\n')


    winsound.Beep(frequency=200, duration=1000)
    plot_accuracy(plt_lists)
