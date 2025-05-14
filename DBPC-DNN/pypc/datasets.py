import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision import datasets, transforms
from pypc import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "./data/mnist"
class MNIST(datasets.MNIST):
    def __init__(self, cf, train):
        scale = cf.label_scale
        size = cf.train_size
        transform = _get_transform(train = train, normalize=cf.normalize, addGaussianNoise=cf.AddGaussianNoise, AddRotaion=cf.AddRotaion, ResizedCrop=cf.ResizedCrop_size, mean=(0.1307), std=(0.3081))
        super().__init__(root = data_dir, download=True, transform=transform, train=train)
        self.scale = scale
        self.expand_size = cf.expand_size
        if size is not None:
            self._reduce(size)

    def expand_lable(self, input, expand_size):
        expand_lable_out = torch.zeros((10*expand_size), ).to(DEVICE)
        for i in range(10):
            if input[i] == 1:  # input 0 digit
                expand_lable_out[i*expand_size : i*expand_size+(expand_size)] = 1
        return expand_lable_out

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        # data = _to_vector(data)
        target = _one_hot(target)
        if self.expand_size > 1:
            target = self.expand_lable(target, self.expand_size)
        if self.scale is not None:
            target = _scale(target, self.scale)
        return data, target

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]


class CIFAR10(datasets.CIFAR10):
    def __init__(self, cf, train):
        scale = cf.label_scale
        size = cf.train_size
        transform = _get_transform_3_channel_color(train=train, normalize=cf.normalize, addGaussianNoise=cf.AddGaussianNoise, AddRotaion=cf.AddRotaion, ResizedCrop=cf.ResizedCrop_size, mean=(0.1307), std=(0.3081))
        super().__init__(root="./data/cifar10", download=True, transform=transform, train=train)
        self.scale = scale
        self.expand_size = cf.expand_size
        if size is not None:
            self._reduce(size)

    def expand_lable(self, input, expand_size):
        expand_lable_out = torch.zeros((10 * expand_size), ).to(DEVICE)
        for i in range(10):
            if input[i] == 1:  # input 0 digit
                expand_lable_out[i * expand_size: i * expand_size + (expand_size)] = 1
        return expand_lable_out

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        # data = _to_vector(data)
        target = _one_hot(target)
        if self.expand_size > 1:
            target = self.expand_lable(target, self.expand_size)
        if self.scale is not None:
            target = _scale(target, self.scale)
        return data, target

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]

class FashionMNIST(datasets.FashionMNIST):
    def __init__(self, cf, train):
        scale = cf.label_scale
        size = cf.train_size
        transform = _get_transform(train = train, normalize=cf.normalize, addGaussianNoise=cf.AddGaussianNoise, AddRotaion=cf.AddRotaion, ResizedCrop=cf.ResizedCrop_size, mean=(0.1307), std=(0.3081))
        super().__init__("./data/FashionMNIST", download=True, transform=transform, train=train)
        self.scale = scale
        self.expand_size = cf.expand_size
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        # data = _to_vector(data)
        target = _one_hot(target)
        if self.expand_size > 1:
            target = self.expand_lable(target, self.expand_size)
        if self.scale is not None:
            target = _scale(target, self.scale)
        return data, target

    def expand_lable(self, input, expand_size):
        expand_lable_out = torch.zeros((10*expand_size), ).to(DEVICE)
        for i in range(10):
            if input[i] == 1:  # input 0 digit
                expand_lable_out[i*expand_size : i*expand_size+(expand_size)] = 1
        return expand_lable_out

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_dataloader(dataset, shuffle, batch_size):
    dataloader = data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=True)
    return list(map(_preprocess_batch, dataloader))

def get_dataloader_v2(dataset, shuffle, batch_size, sampler):
    # dataloader = data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=False, drop_last=True, sampler=sampler)
    return list(map(_preprocess_batch, dataloader))

def accuracy(pred_labels, true_labels):
    batch_size = pred_labels.size(0)
    correct = 0
    for b in range(batch_size):
        if torch.argmax(pred_labels[b, :]) == torch.argmax(true_labels[b, :]):
            correct += 1
    return correct / batch_size


def plot_imgs(img_preds, path):
    imgs = img_preds.cpu().detach().numpy()
    imgs = imgs[0:10, :]
    imgs = [np.reshape(imgs[i, :], [28, 28]) for i in range(imgs.shape[0])]
    _, axes = plt.subplots(2, 5)
    axes = axes.flatten()
    for i, img in enumerate(imgs):
        axes[i].imshow(img, cmap="gray")
    plt.savefig(path)
    plt.close("all")

def _preprocess_batch(batch):
    batch[0] = utils.set_tensor(batch[0])
    batch[1] = utils.set_tensor(batch[1])
    return (batch[0], batch[1])


def _get_transform(train = False, normalize=True, addGaussianNoise=True, AddRotaion=False, ResizedCrop = 28, mean=(0.5), std=(0.5)):
    if train:
        if (ResizedCrop != 28) & addGaussianNoise & AddRotaion & normalize:
            transform = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.Resize(size=ResizedCrop),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation((-10, 10)),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307), std=(0.3081))
                ])
        elif (ResizedCrop != 28) & addGaussianNoise & AddRotaion:
            transform = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.Resize(size=ResizedCrop),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation((-10, 10)),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor()
                ])
        elif (ResizedCrop != 28) & addGaussianNoise:
            transform = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.Resize(size=ResizedCrop),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor()
                ])
        elif (ResizedCrop != 28):
            transform = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.Resize(size=ResizedCrop),
                transforms.ToTensor()
                ])
        elif addGaussianNoise & AddRotaion & normalize:
            transform = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation((-10, 10)),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307), std=(0.3081))
                ])
        elif addGaussianNoise & AddRotaion:
            transform = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation((-10, 10)),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor()
                ])
        elif addGaussianNoise:
            transform = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor()
                ])
        elif AddRotaion & normalize:
            transform = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation((-10, 10)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307), std=(0.3081))
                ])
        else:
            transform = transforms.Compose([
                    # transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.1307), std=(0.3081))
                    ])

    else:
        if (ResizedCrop != 28):
            transform = transforms.Compose([
                    # transforms.Grayscale(3),
                    transforms.Resize(size=ResizedCrop),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.1307), std=(0.3081))
                    ])
        else:
            transform = transforms.Compose([
                    # transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.1307), std=(0.3081))
                    ])
    return transform

def _get_transform_fashion_mnist(train = False, normalize=True, addGaussianNoise=True, AddRotaion=False, ResizedCrop = 28, mean=(0.5), std=(0.5)):
    rotation = 20
    Affine = 0.2
    if train:
        if (ResizedCrop != 28) & addGaussianNoise & AddRotaion & normalize:
            transform = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.Resize(size=ResizedCrop),
                transforms.RandomAffine(degrees=0, translate=(Affine, Affine)),
                transforms.RandomRotation((-rotation, rotation)),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))
                ])
        elif (ResizedCrop != 28) & addGaussianNoise & AddRotaion:
            transform = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.Resize(size=ResizedCrop),
                transforms.RandomAffine(degrees=0, translate=(Affine, Affine)),
                transforms.RandomRotation((-rotation, rotation)),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor()
                ])
        elif (ResizedCrop != 28) & addGaussianNoise:
            transform = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.Resize(size=ResizedCrop),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor()
                ])
        elif (ResizedCrop != 28):
            transform = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.Resize(size=ResizedCrop),
                transforms.ToTensor()
                ])
        elif addGaussianNoise & AddRotaion & normalize:
            transform = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.RandomAffine(degrees=0, translate=(Affine, Affine)),
                transforms.RandomRotation((-rotation, rotation)),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))
                ])
        elif addGaussianNoise & AddRotaion:
            transform = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.RandomAffine(degrees=0, translate=(Affine, Affine)),
                transforms.RandomRotation((-rotation, rotation)),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor()
                ])
        elif addGaussianNoise:
            transform = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor()
                ])
        elif AddRotaion & normalize:
            transform = transforms.Compose([
                # transforms.Grayscale(3),
                transforms.RandomAffine(degrees=0, translate=(Affine, Affine)),
                transforms.RandomRotation((-rotation, rotation)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))
                ])
        else:
            transform = transforms.Compose([
                    # transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5), std=(0.5))
                    ])

    else:
        if (ResizedCrop != 28):
            transform = transforms.Compose([
                    # transforms.Grayscale(3),
                    transforms.Resize(size=ResizedCrop),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5), std=(0.5))
                    ])
        else:
            transform = transforms.Compose([
                    # transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5), std=(0.5))
                    ])
    return transform


def _get_transform_3_channel_color(train = False, normalize=True, addGaussianNoise=True, AddRotaion=False, ResizedCrop = 28, mean=(0.5), std=(0.5)):
    if train:
        if (ResizedCrop != 28) & addGaussianNoise & AddRotaion & normalize:
            transform = transforms.Compose([
                transforms.Resize(size=ResizedCrop),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation((-10, 10)),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                #transforms.Normalize(mean=(0.1307), std=(0.3081))
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        elif (ResizedCrop != 28) & addGaussianNoise & AddRotaion:
            transform = transforms.Compose([
                transforms.Resize(size=ResizedCrop),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation((-10, 10)),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor()
                ])
        elif (ResizedCrop != 28) & addGaussianNoise:
            transform = transforms.Compose([
                transforms.Resize(size=ResizedCrop),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor()
                ])
        elif (ResizedCrop != 28):
            transform = transforms.Compose([
                transforms.Resize(size=ResizedCrop),
                transforms.ToTensor()
                ])
        elif addGaussianNoise & AddRotaion & normalize:
            transform = transforms.Compose([
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation((-10, 10)),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.1307), std=(0.3081))
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        elif addGaussianNoise & AddRotaion:
            transform = transforms.Compose([
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation((-10, 10)),
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor()
                ])
        elif addGaussianNoise:
            transform = transforms.Compose([
                transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                transforms.ToTensor()
                ])
        elif AddRotaion & normalize:
            transform = transforms.Compose([
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation((-10, 10)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.1307), std=(0.3081))
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.1307), std=(0.3081))
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])

    else:
        if (ResizedCrop != 28):
            transform = transforms.Compose([
                    transforms.Resize(size=ResizedCrop),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=(0.1307), std=(0.3081))
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
        else:
            transform = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=(0.1307), std=(0.3081))
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])

    return transform

def _one_hot(labels, n_classes=10):
    arr = torch.eye(n_classes)
    return arr[labels]


def _scale(targets, factor):
    return targets * factor + 0.5 * (1 - factor) * torch.ones_like(targets)


def _to_vector(batch):
    batch_size = batch.size(0)
    return batch.reshape(batch_size, -1).squeeze()
