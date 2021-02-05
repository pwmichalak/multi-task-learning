import torch 
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim 

import os
import sys
import argparse
import random
import numpy as np 
from tqdm import tqdm


class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class MTL(nn.Module):

    def __init__(self) -> None:

        # initialize nn.Module class
        super(MTL, self).__init__()

        # define convolutional layers of model
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        # define adaptive average pooling layer of model
        self.avgpool = nn.AdaptiveAvgPool2d(16)

        # define fully connected layers of model
        self.classifier = nn.Sequential(
            nn.Linear(16 * 16 * 16, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def CIFAR10TrainLoaderHelper():
    """ Helper function for the CIFAR10TrainLoader; this function splits the dataset in two """
    # define classes for reference
    classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

    # define PIL image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    # load training data
    print('\nLoading training set...\n')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=1, shuffle=True)

    # split the dataset in two and renumber the labels so that there are 5 labels per set
    tmpOneData, tmpOneLabels, tmpTwoData, tmpTwoLabels = [], [], [], []
    for tl in tqdm(trainloader):
        if tl[1] < 5: 
            tmpOneData.append(tl[0])
            tmpOneLabels.append(tl[1])
        else: 
            tmpTwoData.append(tl[0])
            tmpTwoLabels.append(tl[1] % 5)

    return tmpOneData, tmpOneLabels, tmpTwoData, tmpTwoLabels, trainloader


def CIFAR10TrainLoader(dataOnePer: float = 0.5, dataTwoPer: float = 0.5):
    """
    Description:
        Function that loads the CIFAR10 dataset and generates two dataloaders for training (one for each multi-task item)

    Arguments:
        dataOnePer - the proportion of samples to take from the first dataset for data complexity measurements during training
        dataTwoPer - the proportion of samples to take from the second dataset for data complexity measurements during training
    
    Return:
        Dataloaders. This is only going to train on half of the dataset so that there are always 25000 training points
    """

    tmpOneData, tmpOneLabels, tmpTwoData, tmpTwoLabels, trainloader = CIFAR10TrainLoaderHelper()

    # set the number of samples in each class
    numSamplesOne = random.sample(list(range(int(len(trainloader) / 2))), int((len(trainloader) / 2) * dataOnePer))
    numSamplesTwo = random.sample(list(range(int(len(trainloader) / 2))), int((len(trainloader) / 2) * dataTwoPer))
    numSamplesOne.sort()
    numSamplesTwo.sort()

    # select the values from the samples index generator
    dataOne, labelsOne = [tmpOneData[i] for i in numSamplesOne], [tmpOneLabels[i] for i in numSamplesOne]
    dataTwo, labelsTwo = [tmpTwoData[i] for i in numSamplesTwo], [tmpTwoLabels[i] for i in numSamplesTwo]

    # make custom datasets 
    datasetOne = CustomDataset(dataOne, labelsOne)
    datasetTwo = CustomDataset(dataTwo, labelsTwo)

    # create dataloaders for the custom datasets
    trainLoaderOne = torch.utils.data.DataLoader(dataset=datasetOne, batch_size=4, shuffle=True, drop_last=True)
    trainLoaderTwo = torch.utils.data.DataLoader(dataset=datasetTwo, batch_size=4, shuffle=True, drop_last=True)
    print('\nTraining data loaded.')

    # return
    return trainLoaderOne, trainLoaderTwo


def CIFAR10TestLoader():
    """ Function that loads the CIFAR10 dataset and generates two dataloaders for testing (one for each multi-task item) """

    # define classes for reference
    classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

    # define PIL image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    # load testing data
    print('\nLoading test set...\n')
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False)

    # split the testset in two and renumber the labels so that there are 5 labels per set
    testLoaderOne, testLoaderTwo = [], []
    for tl in testloader:
        if tl[1] < 5: testLoaderOne.append(tl)
        else: testLoaderTwo.append([tl[0], tl[1] % 5])

    print('\nTest set loaded.')

    # return
    return testLoaderOne, testLoaderTwo


def trainIter(trainloader, model, optimizer, criterion, epochs):
    """ 
    Description:
        Helper function to train the multi-task learning model with a given trainloader object 
    
    Arguments:
        trainloader - trainloader object containing the training data
        optimizer - torch.optim object for optimization
        model - instance of the Neural Network model class
        criterion - loss function
        epochs - number of training epochs over all of the data
    
    Return:
        returns the model trained on the data from trainloader and the empirical error
    """
    # initalize variables to find the best model
    minModel, minError, minErrorTensor = None, sys.float_info.max - 1, None

    # training loop
    for epoch in range(epochs):

        # loop through dataset
        error = 0.0
        for data in trainloader:

            # data contains labels and images
            images, labels = data
            
            # resize the data because of preprocessing 
            images, labels = torch.squeeze(images), torch.squeeze(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward, backward, optimization
            out = model(images)
            loss = criterion(out, labels)
            error += loss
            loss.backward()
            optimizer.step()

        # get the min error and best model
        if minError > error.detach().numpy():
            minError = error.detach().numpy()
            minErrorTensor = error
            minModel = model
        print('iteration {i}, unnormalized error {e}'.format(i=epoch, e=error))
    
    # return the trained model
    print('Final minimum error: {e}'.format(e=minError))
    return minModel, minErrorTensor / len(trainloader)


def train(dataOnePer: float = 0.5, dataTwoPer: float = 0.5):
    """ 
    Description:
        Function to train the multi-task learning model
    
    Arguments:
        dataOnePer - the proportion of samples to take from the first dataset for data complexity measurements during training
        dataTwoPer - the proportion of samples to take from the second dataset for data complexity measurements during training
    
    Return:
        returns the empirical error from training of both datasets
    """

    # define a MTL model
    mtl = MTL()

    # load data
    tl1, tl2 = CIFAR10TrainLoader(dataOnePer, dataTwoPer)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mtl.parameters(), lr=0.01, momentum=0.9)

    # train the network on the data
    print('\nTraining first classifier...\n')
    mtl, loss1 = trainIter(trainloader=tl1, model=mtl, optimizer=optimizer, criterion=criterion, epochs=10)
    print('\nTraining second classifier...\n')
    mtl, loss2 = trainIter(trainloader=tl2, model=mtl, optimizer=optimizer, criterion=criterion, epochs=10)
    print('\nModel has been trained.\n')
    
    # save the model
    PATH = os.getcwd() + '/mtl_model_{d1}_vs_{d2}.pth'.format(d1=int(dataOnePer*100), d2=int(dataTwoPer*100))
    torch.save(mtl.state_dict(), PATH)

    return loss1.detach().numpy(), loss2.detach().numpy()


def populationError(lossFunc, dataloader, model):
    """
    Description:
        Computes the population loss over the data in the dataloader argument

    Arguments:
        lossFunc - loss function used to compute error
        dataloader - loader object containing data points to use to compute empirical loss
        model - model used to compute the predictions for the data in dataloader

    Returns:
        Population Error value
    """
    error = 0.0
    for data in dataloader:
        img, label = data
        output = model(img)
        loss = lossFunc(output, label)
        error += loss
    return error / len(dataloader)


def accuracy(dataloader, model):
    """
    Description:
        Computes the classification accuracy over the data in the dataloader argument

    Arguments:
        dataloader - loader object containing data points to use to compute empirical loss
        model - model used to compute the predictions for the data in dataloader

    Returns:
        classification accuracy
    """
    correct = 0.0
    total = 0
    i = 0
    for data in dataloader:
        img, label = data
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    return (1.0 * correct) / total


def test(dataOnePer: float = 0.5, dataTwoPer: float = 0.5):
    """
    Description:
        Function to test the multi-task learning model 
    
    Arguments:
        dataOnePer - the proportion of samples to take from the first dataset for data complexity measurements during training
        dataTwoPer - the proportion of samples to take from the second dataset for data complexity measurements during training
    """

    # define path where model is saved
    PATH = os.getcwd() + '/mtl_model_{d1}_vs_{d2}.pth'.format(d1=int(dataOnePer*100), d2=int(dataTwoPer*100))

    # load model
    mtl = MTL()
    mtl.load_state_dict(state_dict=torch.load(PATH))

    # load testing data
    tl1, tl2 = CIFAR10TestLoader()

    # compute generalization error
    print('\nComputing approximate population error...')
    pErr1 = populationError(lossFunc=nn.CrossEntropyLoss(), dataloader=tl1, model=mtl)
    pErr2 = populationError(lossFunc=nn.CrossEntropyLoss(), dataloader=tl2, model=mtl)
    print('Done.')

    return pErr1.detach().numpy(), pErr2.detach().numpy()


def argsParser():
    """ Parse arguments to allow for training and testing modes """
    parser = argparse.ArgumentParser(description="Using Very Deep Autoencoders for Content-Based Image Retrieval")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--test', action='store_true')
    return parser.parse_args()


def main():
    args = argsParser()
    if args.train:
        train()
    elif args.test:
        p1, p2 = test()
        print(p1, p2)
    else:
        print('No flag assigned. Please assign either \'--train\' or \'--eval\'.')

if __name__ == "__main__":
    main()