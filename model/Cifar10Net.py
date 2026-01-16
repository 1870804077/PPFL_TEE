import torch
import torch.nn as nn


class CIFAR10Net(nn.Module):
    def __init__(self, cv1_filters=32, cv5_filters=64, cv6_filters=32, output_size=10):
        super(CIFAR10Net, self).__init__()
        self.cv1_filters = cv1_filters  # 保存参数用于计算展平尺寸
        self.cv6_filters = cv6_filters
        
        # 第1卷积组: 3x3卷积 + BatchNorm + ReLU
        self.conv1 = nn.Conv2d(3, cv1_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(cv1_filters)
        
        # 第2卷积组: 3x3卷积 + BatchNorm + ReLU
        self.conv2 = nn.Conv2d(cv1_filters, cv1_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(cv1_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32→16
        
        # 第3卷积组: 3x3卷积 + BatchNorm + ReLU
        self.conv3 = nn.Conv2d(cv1_filters, cv1_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(cv1_filters)
        
        # 第4卷积组: 3x3卷积 + BatchNorm + ReLU
        self.conv4 = nn.Conv2d(cv1_filters, cv1_filters, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(cv1_filters)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16→8
        
        # 第5卷积组: 3x3卷积 + BatchNorm + ReLU
        self.conv5 = nn.Conv2d(cv1_filters, cv5_filters, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(cv5_filters)
        
        # 第6卷积组: 1x1卷积 + BatchNorm + ReLU
        self.conv6 = nn.Conv2d(cv5_filters, cv6_filters, kernel_size=1, padding=0)
        self.bn6 = nn.BatchNorm2d(cv6_filters)
        
        # 第7卷积组: 1x1卷积 + BatchNorm + ReLU
        self.conv7 = nn.Conv2d(cv6_filters, cv6_filters, kernel_size=1, padding=0)
        self.bn7 = nn.BatchNorm2d(cv6_filters)
        
        # 全连接层：修复展平尺寸计算（cv6_filters * 8 * 8）
        self.fc = nn.Linear(cv6_filters * 8 * 8, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入尺寸: (B, 3, 32, 32) [PyTorch默认NCHW格式]
        
        # 第1组
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 第2组
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool1(x)  # 输出尺寸: (B, cv1_filters, 16, 16)
        
        # 第3组
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # 第4组
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool2(x)  # 输出尺寸: (B, cv1_filters, 8, 8)
        
        # 第5组
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        # 第6组
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        
        # 第7组
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)  # 输出尺寸: (B, cv6_filters, 8, 8)
        
        # 展平：使用动态计算的尺寸，避免硬编码错误
        x = x.view(-1, self.cv6_filters * 8 * 8)
        
        # 全连接输出
        x = self.fc(x)
        return x

    def get_flat_params(self):
        """将模型参数展平为一维向量"""
        return torch.cat([p.flatten().detach() for p in self.parameters()])