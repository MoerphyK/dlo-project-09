import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

def show_img(d):
    #Print (dataset [0] [1]: hier d[1]) # the first dimension is the number of images, the second dimension is 1, and label is returned
    #Print (dataset [0] [0]: hier d[0]) # is 0 and returns picture data
    plt.imshow(d[0])
    plt.title([k for k, v in dataset.class_to_idx.items() if v == d[1]][0])
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    transform = transforms.Resize([200,300])
    dataset = datasets.ImageFolder('../assets/',transform=transform)

    #The picture in cat folder corresponds to label 0 and dog corresponds to 1
    print(dataset.class_to_idx)

    #Paths of all pictures and corresponding labels
    # print(dataset.imgs)

    show_img(dataset[50])
