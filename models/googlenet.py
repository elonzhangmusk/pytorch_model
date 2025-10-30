"""google net in pytorch



[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.

    Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842v1
"""

import torch
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
#  其中，in_channels表示输入通道数，ch1x1表示1x1卷积核的输出通道数，ch3x3red表示3x3卷积核前的1x1卷积核的输出通道数，ch3x3表示3x3卷积核的输出通道数，ch5x5red表示5x5卷积核前的1x1卷积核的输出通道数，ch5x5表示5x5卷积核的输出通道数，pool_proj表示池化层后1x1卷积核的输出通道数。
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),  # 1x1卷积
            nn.BatchNorm2d(ch1x1),  # Batch Normalization
            nn.ReLU(inplace=True)  # ReLU激活函数
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),  # 1x1卷积降维，通道数减少
            nn.BatchNorm2d(ch3x3red),  # Batch Normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),  # 3x3卷积，padding=1，在特征图外围加一圈0从而保持尺寸不变
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )
# 感受野指的是：输出特征图上某个像素点能“看到”输入图像多大区域的信息
# 感受野越大，说明该像素点能获取到的上下文信息越多，有助于捕捉图像中的全局特征
# 这里：感受野 = 3+3-1=5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5, ch5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
# 前向传播函数，将输入数据通过各个分支处理后拼接输出
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogleNet(nn.Module):
    def __init__(self, num_classes=2, aux_logits=True, transform_input=False, init_weights=True):
# num_classes表示分类的类别数，aux_logits表示是否使用辅助分类器，transform_input表示是否对输入进行预处理，init_weights表示是否初始化权重。
        super(GoogleNet, self).__init__()
        self.transform_input = transform_input
# 输出尺寸逐渐减小，通道数增加，为 Inception 模块准备输入。
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
# 整个网络通过多尺度特征提取 + 降采样，实现深层次特征学习。
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
    # Inception 模块：多分支卷积 + 池化
    # 参数解释：
    # Inception(in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj)
    # 每个参数的含义如下：
    # in_channels  : 输入通道数
    # ch1x1        : 第一条分支 1x1 卷积输出通道数
    # ch3x3red     : 第二条分支 1x1 降维通道数（降低计算量）
    # ch3x3        : 第二条分支 3x3 卷积输出通道数
    # ch5x5red     : 第三条分支 1x1 降维通道数
    # ch5x5        : 第三条分支两次 3x3 卷积模拟5x5输出通道数
    # pool_proj    : 第四条分支池化后1x1卷积输出通道数

    # 例子：Inception3a(192, 64, 96, 128, 16, 32, 32)
    # 解释：
    # 192   : 输入通道数 = 上一层输出通道
    # 64    : 分支1 1x1卷积输出64个特征图
    # 96    : 分支2先1x1降维到96通道
    # 128   : 分支2 3x3卷积输出128通道
    # 16    : 分支3先1x1降维到16通道
    # 32    : 分支3连续两次3x3卷积输出32通道
    # 32    : 分支4池化后1x1卷积输出32通道

    # 输出通道 = ch1x1 + ch3x3 + ch5x5 + pool_proj = 64 + 128 + 32 + 32 = 256
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # if self.transform_input:
        #     x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        #     x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        #     x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        #     x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

   # 如果 transform_input=True，则归一化到你数据集的均值和标准差
        if self.transform_input:
            mean = torch.tensor([0.5813, 0.4787, 0.4483], device=x.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.1629, 0.1960, 0.2080], device=x.device).view(1, 3, 1, 1)
            x = (x - mean) / std  # 对每个通道做标准化
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x num_classes
        return x

    def _initialize_weights(self):
        # 遍历网络中所有子模块（Conv2d, Linear, BatchNorm2d 等）
        for m in self.modules():  
            
            # 如果是卷积层或全连接层
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # 使用 Kaiming 正态初始化（He 初始化），适合 ReLU 激活
                # mode='fan_out' 根据输出通道调整方差
                # nonlinearity='relu' 适配ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
                # 如果存在偏置，则初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            # 如果是批归一化层
            elif isinstance(m, nn.BatchNorm2d):
                # BN层权重初始化为1（scale γ）
                nn.init.constant_(m.weight, 1)
                # BN层偏置初始化为0（shift β）
                nn.init.constant_(m.bias, 0)

def googlenet():
    model = GoogleNet()
    state_dict = torch.load('/home/zhanghangning/pytorch/preTrainedCheckpoints/googlenet-1378be20.pth')
    # 只加载匹配的权重
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items()
                       if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model



