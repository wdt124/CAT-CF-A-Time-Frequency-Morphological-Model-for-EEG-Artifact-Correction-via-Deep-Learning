import torch
import torch.nn as nn
import torch.optim as optim
from complexPyTorch.complexLayers import ComplexBatchNorm1d, ComplexConv2d, ComplexLinear,ComplexAvgPool2d,ComplexReLU,ComplexDropout,ComplexBatchNorm2d
class ConvAttentionModel(nn.Module):
    def __init__(self):
        super(ConvAttentionModel, self).__init__()

        # 卷积层：提取特征
        self.conv1d3=nn.Sequential(        
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2)
            )
        
        '''self.conv1d5=nn.Sequential(        
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2)
            )'''
        
        self.conv1d7=nn.Sequential(        
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3),
            
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3),
            
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3),
            
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2)
            )
        self.res=lambda x:x

        self.dropout = nn.Dropout(0.5)
        # 自注意力模块
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=16, batch_first=True,dropout=0)
        # 全连接部分
        self.linear = nn.Sequential(
            nn.Linear(128*64, 64*32),  # 通过全连接层将特征图映射到输入大小
            nn.ReLU(),
            nn.Linear(64*32, 64*16),  # 通过全连接层将特征图映射到输入大小
            nn.ReLU(),            
            nn.Linear(64*16, 512)     
        )
    def forward(self, x):
        batch_size=x.size(0)
        x = x.view(batch_size, 1, -1) 
        x_1=self.conv1d3(x)
        x_2=self.conv1d7(x)
        #

        #x=self.dropout(x)
        x=torch.concatenate([x_1,x_2],dim=1)
        #x=self.dropout(x)
        # Step 3: 注意力
        identity=self.res(x)
        output, attn_weights = self.attention(x, x, x)  # (batch_size, num_segments, embed_dim)
        output=output+identity


        # Step 4: 展平输出
        output = output.reshape(batch_size, 128*64)  # (batch_size, seq_len)
        output=self.linear(output)

        return output
    
class ComplexCNNModel_sep(nn.Module):
    def __init__(self, input_size=512):
        """
        定义一个输入和输出相同形状的CNN网络
        :param input_size: 输入数据的特征维度 (即 512)
        """
        
        super(ComplexCNNModel_sep, self).__init__()
        self.res=lambda x:x
        self.conv_block0_3=nn.Sequential(
            ComplexConv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # 卷积层1
            #ComplexBatchNorm2d(4),
            ComplexReLU(),
            ComplexAvgPool2d(kernel_size=(1,2), stride=2)
            )
        self.conv_block0_5=nn.Sequential(
            ComplexConv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),  # 卷积层1
            #ComplexBatchNorm2d(4),
            ComplexReLU(),
            ComplexAvgPool2d(kernel_size=(1,2), stride=2)
            )
        # 卷积部分
        self.conv_block1 = nn.Sequential(
            ComplexConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # 卷积层1
            #ComplexBatchNorm2d(4),
            ComplexReLU(),
            #ComplexAvgPool2d(kernel_size=(1,2), stride=2),  # 平均池化层
            #ComplexDropout(0.1),
            ComplexConv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),  # 卷积层2
            #ComplexBatchNorm2d(16),
            ComplexReLU(),
            ComplexDropout(0.1)
            #ComplexAvgPool2d(kernel_size=(1,2), stride=2),  # 平均池化层
            #ComplexConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # 卷积层2
            #ComplexBatchNorm2d(16),
            #ComplexReLU(),
            #ComplexAvgPool2d(kernel_size=(1,2), stride=2),  # 平均池化层
            #ComplexDropout(0.1),          
        )
        self.conv_block2 = nn.Sequential(
            #ComplexAvgPool2d(kernel_size=(1,2), stride=2),  # 平均池化层
            #ComplexDropout(0.1),
            ComplexConv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),  # 卷积层2
            ComplexReLU(),
            ComplexConv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2),  # 卷积层2
            ComplexReLU(),
            ComplexDropout(0.1)
        )       
        
        # 全连接部分
        self.fc_layer_1 = nn.Sequential(
            ComplexLinear(16 * input_size, 8 *input_size),  # 通过全连接层将特征图映射到输入大小
            ComplexReLU(),
            ComplexLinear(8 * input_size, 4 *input_size),  # 通过全连接层将特征图映射到输入大小
            ComplexReLU(),
            ComplexLinear(4* input_size, 1 *input_size)
            #ComplexDropout(0.1),
            #ComplexLinear(4 * input_size, 1 *input_size),  # 通过全连接层将特征图映射到输入大小
            #ComplexReLU(),
            #ComplexDropout(0.1),       
        )
        self.fc_layer_2 = nn.Sequential(
            ComplexLinear(16 * input_size, 8 *input_size),  # 通过全连接层将特征图映射到输入大小
            ComplexReLU(),
            ComplexLinear(8 * input_size, 4 *input_size),  # 通过全连接层将特征图映射到输入大小
            ComplexReLU(),
            ComplexLinear(4* input_size, 1 *input_size)
            #ComplexDropout(0.1),
            #ComplexLinear(4 * input_size, 1 *input_size),  # 通过全连接层将特征图映射到输入大小
            #ComplexReLU(),
            #ComplexDropout(0.1),       
        )
        self.fc_layer3=nn.Sequential(
            ComplexLinear(2 * input_size, 1 *input_size),
            ComplexReLU(),
            ComplexLinear(input_size, input_size),
            )  # 通过全连接层将特征图映射到输入大小

        self.mask_layer=nn.Sequential(
            ComplexLinear( input_size, input_size),
            #ComplexReLU(),
            ComplexLinear(input_size, input_size)
            )  # 通过全连接层将特征图映射到输入大小

    def forward(self, input):
        # 假设输入 x 的形状为 (batch_size, 512)，即每条输入是一个一维特征向量
        x=torch.fft.fft(input, dim=-1)
        res_x=self.res(x)
        mask=torch.sigmoid(torch.abs(self.mask_layer(x)))
        x=mask*x+res_x
        x = x.view(x.size(0),1, 1, -1)  # 将数据调整为形状 (batch_size, 1,1, 512)，即单通道
        # 卷积部分
        x_3=self.conv_block0_3(x)
        x_5=self.conv_block0_5(x)
        res_3=self.res(x_3)
        res_5=self.res(x_5)
        conv1 = self.conv_block1(x_3)  # 卷积层处理
        conv2 = self.conv_block2(x_5)
        conv1=conv1+res_3
        conv2=conv2+res_5
        conv1 = conv1.view(conv1.size(0), -1)
        conv2 = conv2.view(conv2.size(0), -1)
        out_1=self.fc_layer_1(conv1)
        out_2=self.fc_layer_2(conv2)
        out=torch.concatenate([out_1,out_2],dim=1)

        # 展平卷积输出，得到适合全连接层的形状
        #out = out.view(out.size(0), -1)  # 展平

        # 全连接层部分
        out = self.fc_layer3(out)  # 通过全连接层映射回输入维度

        return out
class Freq_module_4(nn.Module):
    def __init__(self):
        super(Freq_module_4, self).__init__()
        self.freq_module =ComplexCNNModel_sep()
        self.res=lambda x:x
        self.fc_layers=nn.Sequential(
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
        )
        self.lstm=nn.LSTM(2048,2048,1,batch_first=True)
    def forward(self,time_output,x):
        # 频域模块
        fft_time_output = torch.fft.fft(x, dim=-1)  # 时域预测的傅里叶变换

        # 学习角度信息
        freq_pred = self.freq_module(fft_time_output)
        #freq_res=self.res(fft_time_output)
        # 逆傅里叶变换重建时域信号
        ifft = torch.fft.ifft(freq_pred, dim=-1).real
        ifft_image = torch.fft.ifft(freq_pred, dim=-1).imag
        padding=torch.zeros_like(ifft)

        output=torch.concatenate([time_output,padding,ifft,ifft_image],dim=-1)

        # 时域模块
        output, _ =self.lstm(output)
        reconstructed_signal = self.fc_layers(output)
        return freq_pred,reconstructed_signal

    
class TF_model_4(nn.Module):
    def __init__(self):
        super(TF_model_4, self).__init__()
        self.time_module = ConvAttentionModel()
        self.freq_module =Freq_module_4()
        

    def forward(self, x):
        time_output=self.time_module(x)
        # 学习角度信息
        freq_pred,reconstructed_signal = self.freq_module(time_output,x)
        
        return time_output, freq_pred,reconstructed_signal