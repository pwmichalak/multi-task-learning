import numpy as np
import os
import cv2
import json
import torch
import torchvision

class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

class BaseDataLoader:
    def __init__(self, batch_size=1, train=True, shuffle=True, drop_last=False):
        pass

    def get_loader(self, loader, prob):
        raise NotImplementedError

    def get_labels(self, task):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def num_channels(self):
        raise NotImplementedError

    @property
    def num_classes_single(self):
        raise NotImplementedError

    @property
    def num_classes_multi(self):
        raise NotImplementedError

class CIFAR10Loader(BaseDataLoader):
    def __init__(self, batch_size=128, train=True, shuffle=True, drop_last=False, classes=['cat','dog'], percentages=[1]*10):
        """
        Arguments:
            batch_size: the size of the training batches
            train: 
        """
        # initialize dataset with inputs
        super(CIFAR10Loader, self).__init__(batch_size, train, shuffle, drop_last)

        # The output of torchvision datasets are PILImage images of range [0,1]. Transform them to tensors of normalized range [-1,1].
        # Transforms.Compose basically combines the transforms into one.
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        # load the data
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)

        # define variables with only the classes corresponding to the user-defined input classes 
        self.CIFAR10classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
        self.classesIndex = [i for i in range(10) if self.CIFAR10classes[i] in classes]

        # first obtain a dataloader of all of CIFAR10 so that we can cycle through and get only the pictures with labels in the class
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )
        
        # make a custom dataset containing only the classes contained in the user inputted classes
        images, labels = [], []
        counters = list(range(10))
        for batch_images, batch_labels in dataloader:
            for i, l in zip(batch_images,batch_labels):
                if l in self.classesIndex:

                    # get only 1 / percentages[l] of the datapoints for class l
                    counters[l] += 1
                    if counters[l] % int(1/percentages[l]) == 0:
                        images.append(i)
                        labels.append(l)   

        # update the dataset and the dataloader
        dataset = CustomDataset(data=images, labels=labels)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )
        
        # initialize class variables
        self.task_dataloader = None
        self._len = len(list(dataset))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.classes = classes
        
    def _create_TaskDataLoaders(self):
        """
        Create multi task data loaders for each individual task in the user inputted classes
        """
        images = []
        labels = []

        # load the images and labels with the classes 
        for batch_images, batch_labels in self.dataloader:
            for i, l in zip(batch_images,batch_labels):
                if l in self.classesIndex:
                    images.append(i)
                    labels.append(l)

        # get the data loaders for each individual task
        self.task_dataloader = []
        for t in self.classesIndex:
            dataset = CustomDataset(data=images.copy(), labels=[(c == t).long() for c in labels])
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last
            )
            self.task_dataloader.append(dataloader)

    def get_loader(self, loader='standard', prob='uniform'):
        """
        Get the data loaders (either standard simple task or multitask data loaders)
        """
        # return the standard single task dataloader
        if loader == 'standard':
            return self.dataloader

        # create a multitask dataloader list if it does not exist yet
        if self.task_dataloader is None:
            self._create_TaskDataLoaders()

        # return the multitask dataloaders array
        if loader == 'multi-task':
            return MultiTaskDataLoader(self.task_dataloader, prob)
        else:
            # if the multi-task modle is not joint, return only the task corresponding to the current index (loader is an integer in agents.py)
            assert loader in range(len(self.classes)), 'Unknown loader: {}'.format(loader)
            return self.task_dataloader[loader]

    def get_labels(self, task='standard'):
        """
        Obtain the class labels for the user inputted classes 
        """
        if task == 'standard':
            return self.classesIndex
        else:
            assert task in self.classesIndex, 'Unknown task: {}'.format(task)
            labels = [0 for _ in range(len(self.classes))]
            labels[task] = 1
            return labels

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return self._len

    @property
    def num_channels(self):
        return 3

    @property
    def num_classes_single(self):
        """
        Returns the number of different classes in the classification task if there's only one classifier
        There are a maximum of ten classes (car, plane, dog, etc.). Here, num(classes) = num(tasks)
        """
        return len(self.classes)

    @property
    def num_classes_multi(self):
        """
        Returns a list of the number of classes in the classification task with multiple tasks.
        There are two classes per task (True or False) and at most ten tasks
        """
        return [2 for _ in range(len(self.classes))]

class MultiTaskDataLoader:
    def __init__(self, dataloaders, prob='uniform'):
        self.dataloaders = dataloaders
        self.iters = [iter(loader) for loader in self.dataloaders]

        if prob == 'uniform':
            self.prob = np.ones(len(self.dataloaders)) / len(self.dataloaders)
        else:
            self.prob = prob

        self.size = sum([len(d) for d in self.dataloaders])
        self.step = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.step >= self.size:
            self.step = 0
            raise StopIteration

        task = np.random.choice(list(range(len(self.dataloaders))), p=self.prob)

        try:
            data, labels = self.iters[task].__next__()
        except StopIteration:
            self.iters[task] = iter(self.dataloaders[task])
            data, labels = self.iters[task].__next__()

        self.step += 1

        return data, labels, task