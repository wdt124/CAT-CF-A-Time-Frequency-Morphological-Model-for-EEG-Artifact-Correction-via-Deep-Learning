from models_arch import *
import numpy as np
# 2. 定义CNN模型
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
    
class ComplexCNNModel_deri(nn.Module):
    def __init__(self, input_size=512):
        """
        定义一个输入和输出相同形状的CNN网络
        :param input_size: 输入数据的特征维度 (即 512)
        """
        
        super(ComplexCNNModel_deri, self).__init__()
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
            #ComplexDropout(0.2)
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
            #ComplexDropout(0.2)
        )       
        
        # 全连接部分
        self.fc_layer_1 = nn.Sequential(
            ComplexLinear(16 * input_size, 8 *input_size),  # 通过全连接层将特征图映射到输入大小
            #ComplexReLU(),
            #ComplexLinear(8 * input_size, 4 *input_size),  # 通过全连接层将特征图映射到输入大小
            ComplexReLU(),
            ComplexLinear(8* input_size, 1 *input_size)
            #ComplexDropout(0.1),
            #ComplexLinear(4 * input_size, 1 *input_size),  # 通过全连接层将特征图映射到输入大小
            #ComplexReLU(),
            #ComplexDropout(0.1),       
        )
        self.fc_layer_2 = nn.Sequential(
            ComplexLinear(16 * input_size, 8 *input_size),  # 通过全连接层将特征图映射到输入大小
            ComplexReLU(),
            #ComplexLinear(8 * input_size, 4 *input_size),  # 通过全连接层将特征图映射到输入大小
            #ComplexReLU(),
            ComplexLinear(8* input_size, 1 *input_size)
            #ComplexDropout(0.1),
            #ComplexLinear(4 * input_size, 1 *input_size),  # 通过全连接层将特征图映射到输入大小
            #ComplexReLU(),
            #ComplexDropout(0.1),       
        )
        self.fc_layer3=ComplexLinear(2 * input_size, 1 *input_size) # 通过全连接层将特征图映射到输入大小

        self.fc_layer4=nn.Sequential(
            ComplexLinear(input_size, input_size))  # 通过全连接层将特征图映射到输入大小
            #ComplexLinear(8 * input_size, 4 *input_size),  # 通过全连接层将特征图映射到输入大小
            #ComplexReLU(),
            #ComplexLinear(input_size, input_size))

        self.mask_layer=nn.Sequential(
            ComplexLinear(input_size, input_size),  # 通过全连接层将特征图映射到输入大小
            #ComplexReLU(),
            #ComplexLinear(8 * input_size, 4 *input_size),  # 通过全连接层将特征图映射到输入大小
            #ComplexReLU(),
            ComplexLinear(input_size, input_size))

    def forward(self, input):
        # 假设输入 x 的形状为 (batch_size, 512)，即每条输入是一个一维特征向量
        device=input.device
        x=torch.fft.fft(input, dim=-1)
        #微分
        # 构建频率轴
        N=x.shape[-1]
        fs=256
        freqs = torch.fft.fftfreq(N, d=1/fs)  # shape: (512,)

        # 构建 j2pi f 向量
        j2pi_f = (1j * 2 * torch.pi * freqs).to(device)  # shape: (512,)

        # 广播乘法：自动在每一条信号上应用
        x= x * j2pi_f  # shape: (64, 512)
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
        # 反差分
        freqs_safe = freqs.clone()
        freqs_safe[freqs_safe == 0] = 1.0  # 防止 1/0
        j2pi_1_f = (1/(1j * 2 * torch.pi*freqs_safe)).to(device)  # shape: (512,)

        # 广播乘法：自动在每一条信号上应用
        out= out * j2pi_1_f  # shape: (64, 512)
        out=self.fc_layer4(out)
        return out

class fusion_module(nn.Module):
    def __init__(self):
        super(fusion_module, self).__init__()
        self.fc_layers=nn.Sequential(
            nn.Linear(2048,4096),
            nn.ReLU(),
            nn.Linear(4096,1024),
            nn.ReLU(),
            nn.Linear(1024,512)
        )
        self.lstm=nn.LSTM(2048,2048,1,batch_first=True)
    def forward(self,time_output,freq_output):
        # 融合模块

        ifft = torch.fft.ifft(freq_output, dim=-1).real
        ifft_image = torch.fft.ifft(freq_output, dim=-1).imag
        padding=torch.zeros_like(ifft)

        output=torch.concatenate([time_output,padding,ifft,ifft_image],dim=-1)

        # 时域模块
        output, _ =self.lstm(output)
        reconstructed_signal = self.fc_layers(output)
        return reconstructed_signal
    
class fusion_module_ab(nn.Module):
    def __init__(self):
        super(fusion_module_ab, self).__init__()
        self.fusion_layer=fusion_module()
    def forward(self,x):
        # 融合模块
        time_output=x
        freq_output=torch.fft.fft(x, dim=-1)
        reconstructed_signal=self.fusion_layer(time_output,freq_output)
        return reconstructed_signal

class Freq_module_4(nn.Module):
    def __init__(self):
        super(Freq_module_4, self).__init__()
        self.freq_module =ComplexCNNModel_sep()
        self.res=lambda x:x
        '''self.fc_layers=nn.Sequential(
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
        )
        self.lstm=nn.LSTM(2048,2048,1,batch_first=True)'''
    def forward(self,time_output,x):
        # 频域模块
        #fft_time_output = torch.fft.fft(x, dim=-1)  # 时域预测的傅里叶变换

        # 学习角度信息
        #老版本：
        #freq_pred = self.freq_module(time_output)
        freq_pred = self.freq_module(x)
        #freq_res=self.res(fft_time_output)
        # 逆傅里叶变换重建时域信号
        '''ifft = torch.fft.ifft(freq_pred, dim=-1).real
        ifft_image = torch.fft.ifft(freq_pred, dim=-1).imag
        padding=torch.zeros_like(ifft)

        output=torch.concatenate([time_output,padding,ifft,ifft_image],dim=-1)

        # 时域模块
        output, _ =self.lstm(output)
        reconstructed_signal = self.fc_layers(output)'''
        return freq_pred

    
class TF_model_4(nn.Module):
    def __init__(self):
        super(TF_model_4, self).__init__()
        self.time_module = ConvAttentionModel()
        self.freq_module =Freq_module_4()
        self.fusion_module=fusion_module()
        

    def forward(self, x):
        time_output=self.time_module(x)
        # 学习角度信息
        freq_pred= self.freq_module(time_output,x)
        reconstructed_signal=self.fusion_module(time_output,freq_pred)

        return time_output,freq_pred,reconstructed_signal

class Freq_module_5(nn.Module):
    def __init__(self):
        super(Freq_module_5, self).__init__()
        self.freq_module =ComplexCNNModel_deri()
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
    
class TF_model_5(nn.Module):
    def __init__(self):
        super(TF_model_5, self).__init__()
        self.time_module = ConvAttentionModel()
        self.freq_module =Freq_module_5()
        

    def forward(self, x):
        time_output=self.time_module(x)
        # 学习角度信息
        freq_pred,reconstructed_signal = self.freq_module(time_output,x)
        
        return time_output, freq_pred,reconstructed_signal