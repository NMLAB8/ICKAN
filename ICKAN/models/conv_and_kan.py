from torch import nn
import sys
import torch.nn.functional as F
import torch


from .KANLinear import KANLinear

class ICKAN(nn.Module):
    def __init__(self):
        super(ICKAN, self).__init__()
        # Convolutional layer, assuming an input with 1 channel (grayscale image)
        # and producing 16 output channels, with a kernel size of 3x3
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)

        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Flatten layer
        self.flatten = nn.Flatten()

        # KAN layer
        fc_input_size = self._get_fc_input_features(64)
        self.kan1 = KANLinear(
            8560,
            1000,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1])
        
        self.kan2 = KANLinear(
            1000,
            20,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1])

    def _get_fc_input_features(self, n_mels):
        # 使用一个虚拟输入来计算特征数
        device = next(self.conv1.parameters())[0].device
        x = torch.randn(1, 1, n_mels, 431, device=device)  # 更新输入大小
        print('input:', '\t\t', x.shape)
        x = self.conv1(x)
        print('conv1:', '\t\t', x.shape)
        x = self.maxpool(x)
        print('maxpool:', '\t\t', x.shape)
        
        x = self.conv2(x)
        print('conv2:', '\t\t', x.shape)
        x = self.maxpool(x)
        print('maxpool:', '\t\t', x.shape)
    
        x = torch.flatten(x, 1)
        print('flatten:', '\t\t', x.shape)
        return x.size(1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.kan1(x)
        x = self.kan2(x)
        x = F.log_softmax(x, dim=1)

        return x