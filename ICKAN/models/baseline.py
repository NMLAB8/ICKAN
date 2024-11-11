from torch import nn
import torch
import torch.nn.functional as F
import sys
# directory reach
# sys.path.append('../kan_convolutional')

class ConvComp(nn.Module):
    def __init__(self):
        super(ConvComp, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(64, affine=True, track_running_stats=True)

        
        self.conv2 = nn.Conv2d(64, 32, kernel_size=2, padding='same')
        self.bn2 = nn.BatchNorm2d(32, affine=True, track_running_stats=True)

        self.dropout = nn.Dropout(0.5)
        self.mean_pooler =  nn.AvgPool2d(kernel_size=2)

        n_fc_input = self._get_fc_input_features(64)
        self.fc1 = nn.Linear(n_fc_input, 128) 
        self.fc2 = nn.Linear(128, 20)


    def _get_fc_input_features(self, n_mels):
        # 使用一个虚拟输入来计算特征数
        
        device = next(self.conv1.parameters())[0].device
        x = torch.randn(1, 1, n_mels, 431, device=device)  # 更新输入大小
        print('input:', '\t\t', x.shape)
        x = self.conv1(x)
        print('conv1:', '\t\t', x.shape)
        x = self.mean_pooler(x)
        print('mean_pooler:', '\t\t', x.shape)
        x = self.bn1(x)     
        print('bn1:', '\t\t', x.shape)
        

        x = self.conv2(x)
        print('conv2:', '\t\t', x.shape)
        x = self.mean_pooler(x)
        print('mean_pooler:', '\t\t', x.shape)
        x = self.bn2(x)
        print('bn2:', '\t\t', x.shape)

        x = torch.flatten(x, 1)
        print('flatten:', '\t\t', x.shape)
        return x.size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mean_pooler(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.mean_pooler(x)
        x = self.bn2(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)