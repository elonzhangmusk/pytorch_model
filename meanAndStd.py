import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据集根目录
data_root = '/home/zhanghangning/pytorch/left-eye'

# 选择 train 或 test
split = 'train'  # 或 'train'

transform = transforms.Compose([
    transforms.ToTensor()  # 转为 [0,1] 张量
])

# 使用 ImageFolder 自动读取子文件夹分类
dataset = datasets.ImageFolder(root=f'{data_root}/{split}', transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

mean = 0.
std = 0.
nb_samples = 0

for data, _ in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)  # flatten H*W
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print('Mean:', mean)
print('Std:', std)
