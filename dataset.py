""" train and test dataset

author baiyu
"""
import os
import sys
import pickle

from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch


class LeftEyeDataset(Dataset):
    """自定义 left-eye 数据集"""

    def __init__(self, root_dir, transform=None):
        """
        root_dir: 数据根目录，例如 "/home/zhanghangning/pytorch/left-eye/train"
        transform: torchvision.transforms 预处理
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(cls_folder, fname)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_training_dataloader_TCMEyes(mean, std, batch_size=128, num_workers=4, shuffle=True):
    """返回 left-eye 训练集 DataLoader"""
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = LeftEyeDataset(settings.LEFTEYE_TRAIN_PATH, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader


def get_test_dataloader_TCMEyes(mean, std, batch_size=128, num_workers=4, shuffle=False):
    """返回 left-eye 测试集 DataLoader"""
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = LeftEyeDataset(settings.LEFTEYE_TEST_PATH, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return test_loader



# class CIFAR100Train(Dataset):
#     """cifar100 test dataset, derived from
#     torch.utils.data.DataSet
#     """

#     def __init__(self, path, transform=None):
#         #if transform is given, we transoform data using
#         with open(os.path.join(path, 'train'), 'rb') as cifar100:
#             self.data = pickle.load(cifar100, encoding='bytes')
#         self.transform = transform

#     def __len__(self):
#         return len(self.data['fine_labels'.encode()])

#     def __getitem__(self, index):
#         label = self.data['fine_labels'.encode()][index]
#         r = self.data['data'.encode()][index, :1024].reshape(32, 32)
#         g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
#         b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
#         image = numpy.dstack((r, g, b))

#         if self.transform:
#             image = self.transform(image)
#         return label, image

# class CIFAR100Test(Dataset):
#     """cifar100 test dataset, derived from
#     torch.utils.data.DataSet
#     """

#     def __init__(self, path, transform=None):
#         with open(os.path.join(path, 'test'), 'rb') as cifar100:
#             self.data = pickle.load(cifar100, encoding='bytes')
#         self.transform = transform

#     def __len__(self):
#         return len(self.data['data'.encode()])

#     def __getitem__(self, index):
#         label = self.data['fine_labels'.encode()][index]
#         r = self.data['data'.encode()][index, :1024].reshape(32, 32)
#         g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
#         b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
#         image = numpy.dstack((r, g, b))

#         if self.transform:
#             image = self.transform(image)
#         return label, image

