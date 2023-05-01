from PIL import Image
import os.path
import torch
import warnings
import torch.utils.data as data


class FEMNIST(data.Dataset):

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, args, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            self.data, self.targets = self.generate_ds(args, self.root)
        else:
            self.data, self.targets = self.generate_ds_test(args, self.root)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.open(img).convert('L')
        img.thumbnail((28, 28), Image.ANTIALIAS)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'data', 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def generate_ds(self, args, root):
        num_class = args.num_classes
        num_img = 2500
        data = []
        targets = torch.zeros([num_class  * num_img], dtype=int)
        files = os.listdir(os.path.join(root, 'by_class'))
        for i in range(num_class):
            for k in range(num_img):
                img = os.path.join(root, 'by_class', files[i], 'train_' + files[i], 'train_' + files[i] + '_'+str("%05d"%k)+'.png')
                data.append(img)
                targets[i * num_img + k] = i
        targets = targets.reshape([num_class * num_img])
        return data, targets

    def generate_ds_test(self, args, root):
        num_class = args.num_classes
        num_img = 500
        data = []
        targets = torch.zeros([num_class * num_img], dtype=int)
        files = os.listdir(os.path.join(root, 'by_class'))
        for i in range(num_class):
            for k in range(num_img):
                img = os.path.join(root, 'by_class', files[i], 'hsf_0', 'hsf_0'+'_'+str("%05d"%(k))+'.png')
                data.append(img)
                targets[i * num_img + k] = i
        targets = targets.reshape([num_class * num_img])
        return data, targets
        