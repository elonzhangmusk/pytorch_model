import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1️⃣ 定义数据增强和预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪（增强）
    transforms.RandomHorizontalFlip(),     # 随机翻转
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),   # CIFAR100 均值
        (0.2675, 0.2565, 0.2761)    # CIFAR100 方差
    ),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    ),
])

# 2️⃣ 下载并加载数据集
trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test
)

# 3️⃣ 用 DataLoader 封装成批次
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 4️⃣ 测试是否能取出一个 batch
images, labels = next(iter(trainloader))
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
