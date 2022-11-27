# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

import os
from typing import *

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import numpy as np
from numpy.random import default_rng

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"
DATASET_LOC = './dataset_cache'

# list of all datasets
DATASETS = ["imagenet", "cifar10", "cifar100", "mnist", "fmnist"]


class MNIST_Targeted(datasets.MNIST):
    def __init__(self, root, prob_path, *args, **kwargs):
        super(MNIST_Targeted, self).__init__(root, *args, **kwargs)
        self.prob_path = prob_path
        self.probs = torch.from_numpy(np.load(prob_path))

    def __getitem__(self, index):
        img, target = super(MNIST_Targeted, self).__getitem__(index)
        prob = self.probs[index].float()
        return img, target, prob

class FMNIST_Targeted(datasets.FashionMNIST):
    def __init__(self, root, prob_path, *args, **kwargs):
        super(FMNIST_Targeted, self).__init__(root, *args, **kwargs)
        self.prob_path = prob_path
        self.probs = torch.from_numpy(np.load(prob_path))

    def __getitem__(self, index):
        img, target = super(FMNIST_Targeted, self).__getitem__(index)
        prob = self.probs[index].float()
        return img, target, prob



class CIFAR10_Targeted(datasets.CIFAR10):
    def __init__(self, root, prob_path, *args, **kwargs):
        super(CIFAR10_Targeted, self).__init__(root, *args, **kwargs)
        self.prob_path = prob_path
        self.probs = torch.from_numpy(np.load(prob_path))

    def __getitem__(self, index):
        img, target = super(CIFAR10_Targeted, self).__getitem__(index)
        prob = self.probs[index].float()
        return img, target, prob

class CIFAR100_Targeted(datasets.CIFAR100):
    def __init__(self, root, prob_path, *args, **kwargs):
        super(CIFAR100_Targeted, self).__init__(root, *args, **kwargs)
        self.prob_path = prob_path
        self.probs = torch.from_numpy(np.load(prob_path))

    def __getitem__(self, index):
        img, target = super(CIFAR100_Targeted, self).__getitem__(index)
        prob = self.probs[index].float()
        return img, target, prob


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar10":
        return _cifar10(split)
    elif dataset == "cifar100":
        return _cifar100(split)
    elif dataset == "mnist":
        return _mnist(split)
    elif dataset == "fmnist":
        return _fmnist(split)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "cifar100":
        return 100
    elif dataset == "mnist":
        return 10
    elif dataset == "fmnist":
        return 10


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "cifar100":
        return NormalizeLayer(_CIFAR100_MEAN, _CIFAR100_STDDEV)
    elif dataset == "mnist":
        return torch.nn.Identity()
    elif dataset == "fmnist":
        return torch.nn.Identity()



_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
_CIFAR100_STDDEV = [0.2675, 0.2565, 0.2761]

def _mnist(split: str) -> Dataset:
    if split == "train":
        return datasets.MNIST(DATASET_LOC, train=True, download=True, transform=transforms.ToTensor())
    elif split == "train_t0.25":
        return MNIST_Targeted(DATASET_LOC, 'test/smooth_prediction/mnist/cohen/0/model_conf_0.25_train.npy',
                              train=True, download=True, transform=transforms.ToTensor())
    elif split == "train_t0.5":
        return MNIST_Targeted(DATASET_LOC, 'test/smooth_prediction/mnist/cohen/0/model_conf_0.25_train.npy',
                              train=True, download=True, transform=transforms.ToTensor())
    elif split == "train_t1.0":
        return MNIST_Targeted(DATASET_LOC, 'test/smooth_prediction/mnist/cohen/0/model_conf_0.25_train.npy',
                              train=True, download=True, transform=transforms.ToTensor())
    elif split == "test":
        return datasets.MNIST(DATASET_LOC, train=False, transform=transforms.ToTensor())

def _fmnist(split: str) -> Dataset:
    if split == "train":
        return datasets.FashionMNIST(DATASET_LOC, train=True, download=True, transform=transforms.ToTensor())
    elif split == "test":
        return datasets.FashionMNIST(DATASET_LOC, train=False, transform=transforms.ToTensor())
    elif split == "train_t0.25":
        return FMNIST_Targeted(DATASET_LOC, 'test/smooth_prediction/fmnist/0/model_conf_0.25_train.npy',
                              train=True, download=True, transform=transforms.ToTensor())
    elif split == "train_t0.5":
        return FMNIST_Targeted(DATASET_LOC, 'test/smooth_prediction/fmnist/0/model_conf_0.25_train.npy',
                              train=True, download=True, transform=transforms.ToTensor())
    elif split == "train_t1.0":
        return FMNIST_Targeted(DATASET_LOC, 'test/smooth_prediction/fmnist/0/model_conf_0.25_train.npy',
                              train=True, download=True, transform=transforms.ToTensor())

   
def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10(DATASET_LOC, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "train_t0.25":
        return CIFAR10_Targeted(DATASET_LOC, 'test/smooth_prediction/cifar10/cohen/0/model_conf_0.25_train.npy',train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "train_t0.5":
        return CIFAR10_Targeted(DATASET_LOC, 'test/smooth_prediction/cifar10/cohen/0/model_conf_0.25_train.npy',train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "train_t1.0":
        return CIFAR10_Targeted(DATASET_LOC, 'test/smooth_prediction/cifar10/cohen/0/model_conf_0.25_train.npy',train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10(DATASET_LOC, train=False, download=True, transform=transforms.ToTensor())


def _cifar100(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR100(DATASET_LOC, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "train_t0.25":
        return CIFAR100_Targeted(DATASET_LOC, 'test/smooth_prediction/cifar100/cohen/0/model_conf_0.25_train.npy',train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "train_t0.5":
        return CIFAR100_Targeted(DATASET_LOC, 'test/smooth_prediction/cifar100/cohen/0/model_conf_0.25_train.npy',train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "train_t1.0":
        return CIFAR100_Targeted(DATASET_LOC, 'test/smooth_prediction/cifar100/cohen/0/model_conf_0.25_train.npy',train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR100(DATASET_LOC, train=False, download=True, transform=transforms.ToTensor())




class ImageNet_Targeted(datasets.ImageFolder):
        def __init__(self, root, prob_path, *args, **kwargs):
            super(ImageNet_Targeted, self).__init__(root, *args, **kwargs)
            self.prob_path = prob_path
            self.probs = torch.from_numpy(np.load(prob_path))

        def __getitem__(self, index):
            img, target = super(ImageNet_Targeted, self).__getitem__(index)
            prob = self.probs[index].float()
            return img, target, prob

def _imagenet(split: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    elif split == "train_t1.0":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        return ImageNet_Targeted(subdir, "test/smooth_prediction/imagenet/cohen/0/model_conf_0.25_train.npy", transform)
    return datasets.ImageFolder(subdir, transform)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).to(device)
        self.sds = torch.tensor(sds).to(device)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
