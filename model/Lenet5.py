import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.avgpool1(x)
        x = self.tanh(self.conv2(x))
        x = self.avgpool2(x)
        x = self.tanh(self.conv3(x))
        x = x.view(-1, 120)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_flat_params(self):
        return torch.cat([p.flatten().detach() for p in self.parameters()])

