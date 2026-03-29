import torch
import torch.nn as nn
import torch.nn.functional as F
# 2. 定义CNN模型
class vanilla_CNNModel(nn.Module):
    def __init__(self, input_size=512):
        """
        定义一个输入和输出相同形状的CNN网络
        :param input_size: 输入数据的特征维度 (即 512)
        """
        super(vanilla_CNNModel, self).__init__()

        # 卷积部分
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),  # 卷积层1
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),  # 卷积层2
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),  # 卷积层3
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        # 全连接部分
        self.fc_layers = nn.Sequential(
            nn.Linear(8 * input_size, input_size)  # 通过全连接层将特征图映射到输入大小
        )

    def forward(self, x):
        # 假设输入 x 的形状为 (batch_size, 512)，即每条输入是一个一维特征向量
        x = x.view(x.size(0), 1, -1)  # 将数据调整为形状 (batch_size, 1, 512)，即单通道

        # 卷积部分
        x = self.conv_layers(x)  # 卷积层处理

        # 展平卷积输出，得到适合全连接层的形状
        x = x.view(x.size(0), -1)  # 展平

        # 全连接层部分
        x = self.fc_layers(x)  # 通过全连接层映射回输入维度

        return x
    

# 2. 定义CNN模型
class vanilla_RNNModel(nn.Module):
    def __init__(self):
        """
        定义一个输入和输出相同形状的CNN网络
        :param input_size: 输入数据的特征维度 (即 512)
        """
        super(vanilla_RNNModel, self).__init__()

        # 卷积部分
        self.lstm=nn.LSTM(512,512,1,batch_first=True)
        self.fc_1=nn.Linear(512,1024)
        self.relu=nn.ReLU()
        self.fc_2=nn.Linear(1024,512)

    def forward(self, x):
        # 卷积部分
        x,_ = self.lstm(x)  # 卷积层处理
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        return x
    
# 定义一个基本的 FCNN 结构
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(512, 1024)  # 第一层全连接
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.fc2 = nn.Linear(1024, 1024)  # 输出层
        self.fc3 = nn.Linear(1024, 512)  # 输出层

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x