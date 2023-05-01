
from http import client
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from utils.sampling import pathological, pathological_lt, practical, practical_lt, from_json
import os, json
from utils.femnist import FEMNIST


# user_groups = { key：客户端id；value：随机选择的样本id集合 }
def get_dataset(args):
    data_dir = args.data_dir + args.dataset
    if args.dataset == 'mnist':
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))
    elif args.dataset == 'fashion':
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))
    elif args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    elif args.dataset == 'femnist':
        train_dataset = FEMNIST(args, data_dir, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))
        test_dataset = FEMNIST(args, data_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))
    elif args.dataset.startswith('synthetic'):
        train_dataset, test_dataset = {}, {}
        train_file_path = os.path.join(data_dir, "train.json")
        with open(train_file_path, 'r') as inf:
            cdata = json.load(inf)
        train_dataset.update(cdata['user_data'])
        test_file_path = os.path.join(data_dir, "test.json")
        with open(test_file_path, 'r') as inf:
            cdata = json.load(inf)
        test_dataset.update(cdata['user_data'])
    # ----------------------------------------------- 划分方法
    if args.sampling == "pathological":
        train_groups, classes_list, label_counts = pathological(train_dataset, args.seed, args.num_classes, args.num_users, args.n)
        test_groups = pathological_lt(test_dataset, args.num_classes, args.num_users, args.n, classes_list)
    elif args.sampling == "practical":
        train_groups, label_counts = practical(train_dataset, args.num_users, args.num_classes)
        test_groups = practical_lt(test_dataset, args.num_users, args.num_classes, label_counts)
    elif args.sampling == "fromJson":
        train_dataset, train_groups, label_counts = from_json(train_dataset, args.num_users, args.num_classes)
        test_dataset, test_groups, _ = from_json(test_dataset, args.num_users, args.num_classes)
    return train_dataset, test_dataset, train_groups, test_groups, label_counts


def read_data(args):
    clients = []
    train_dataset, test_dataset, train_groups, test_groups, label_counts = get_dataset(args)
    for user_id, idx in train_groups.items():
        clients.append(user_id)
    return clients, train_dataset, test_dataset, train_groups, test_groups, label_counts


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
