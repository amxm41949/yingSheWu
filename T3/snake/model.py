import torch
import torch.nn as nn
import torch.nn.functional as F

class SnakeNet(nn.Module):
    def __init__(self, board_size, num_actions=4):
        super().__init__()
        self.board_size = board_size
        self.num_actions = num_actions
        input_channels =  3 # self, others, fruit
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(32 * board_size * board_size, 64)
        
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)
    
class SnakeMLP(nn.Module):
    def __init__(self, board_size, num_actions=4):
        super(SnakeMLP, self).__init__()
        self.board_size = board_size
        input_dim = 3 * board_size * board_size  # 3个通道
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_actions)
    
    def forward(self, x):
        # 输入 x 的形状为 [batch, channels, height, width]
        x = x.view(x.size(0), -1)  # 展平所有输入特征
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)