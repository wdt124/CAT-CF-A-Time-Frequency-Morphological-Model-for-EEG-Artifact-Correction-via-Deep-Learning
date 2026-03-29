import torch
import torch.nn as nn
import torch.optim as optim
from complexPyTorch.complexLayers import ComplexBatchNorm1d, ComplexConv2d, ComplexLinear,ComplexAvgPool2d,ComplexReLU,ComplexDropout,ComplexBatchNorm2d

# 2. 定义CNN模型
class ComplexCNNModel(nn.Module):
    def __init__(self, input_size=512):
        """
        定义一个输入和输出相同形状的CNN网络
        :param input_size: 输入数据的特征维度 (即 512)
        """
        
        super(ComplexCNNModel, self).__init__()
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



    def forward(self, x):
        # 假设输入 x 的形状为 (batch_size, 512)，即每条输入是一个一维特征向量
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


# 2. 定义CNN模型
class CNNModel(nn.Module):
    def __init__(self, input_size=512):
        """
        定义一个输入和输出相同形状的CNN网络
        :param input_size: 输入数据的特征维度 (即 512)
        """
        super(CNNModel, self).__init__()

        # 卷积部分
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),  # 卷积层1
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),  # 卷积层2
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),  # 卷积层3
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # 全连接部分
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * input_size, input_size)  # 通过全连接层将特征图映射到输入大小
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
class LittleCNNModel(nn.Module):
    def __init__(self, input_size=512):
        """
        定义一个输入和输出相同形状的CNN网络
        :param input_size: 输入数据的特征维度 (即 512)
        """
        super(LittleCNNModel, self).__init__()
        self.conv_layers_3_0=nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # 卷积层1
            #nn.BatchNorm1d(4),
            nn.ReLU(),
            )
        self.conv_layers_5_0=nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),  # 卷积层1
            #nn.BatchNorm1d(4),
            nn.ReLU(),
            )
        # 卷积部分
        self.conv_layers_3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # 卷积层1
            #nn.BatchNorm1d(4),
            nn.ReLU(),
            #nn.AvgPool1d(kernel_size=2, stride=2),  # 平均池化层
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),  # 卷积层2
            #nn.BatchNorm1d(16),
            nn.ReLU(),
            #nn.AvgPool1d(kernel_size=2, stride=2),  # 平均池化层
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # 卷积层2
            #nn.BatchNorm1d(16),
            nn.ReLU(),
            #nn.AvgPool1d(kernel_size=2, stride=2),  # 平均池化层
            #nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 卷积层2
            #nn.BatchNorm1d(16),
            #nn.ReLU(),
            #nn.AvgPool1d(kernel_size=2, stride=2),  # 平均池化层
            #nn.Dropout(0.4)
        )
        self.conv_layers_5 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),  # 卷积层1
            #nn.BatchNorm1d(4),
            nn.ReLU(),
            #nn.AvgPool1d(kernel_size=2, stride=2),  # 平均池化层
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2),  # 卷积层2
            #nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            #nn.AvgPool1d(kernel_size=2, stride=2),  # 平均池化层
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),  # 卷积层2
            #nn.BatchNorm1d(16),
            nn.ReLU(),
            #nn.AvgPool1d(kernel_size=2, stride=2),  # 平均池化层
            #nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),  # 卷积层2
            #nn.BatchNorm1d(16),
            #nn.ReLU(),
            #nn.AvgPool1d(kernel_size=2, stride=2),  # 平均池化层
            #nn.Dropout(0.4)
        )



        # 全连接部分
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * input_size, 8 *input_size),  # 通过全连接层将特征图映射到输入大小
            nn.ReLU(),
            #nn.Dropout(0.2),
            #nn.Linear(8 * input_size, 2 *input_size),  # 通过全连接层将特征图映射到输入大小
            #nn.ReLU(),
            nn.Linear(8 * input_size, 1 *input_size),  # 通过全连接层将特征图映射到输入大小
            #nn.ReLU()
        )

        self.res=lambda x:x
        self.lstm=nn.LSTM(8,8,1,batch_first=True)

    def forward(self, x):
        # 假设输入 x 的形状为 (batch_size, 512)，即每条输入是一个一维特征向量
        x = x.view(x.size(0), 1, -1)  # 将数据调整为形状 (batch_size, 1, 512)，即单通道
        # 卷积部分
        x_3_0 = self.conv_layers_3_0(x)  # 卷积层处理
        x_5_0 =self.conv_layers_5_0(x)
        x_3_res=self.res(x_3_0)
        x_5_res=self.res(x_5_0)

        x_3=self.conv_layers_3(x_3_0)+x_3_res
        x_5=self.conv_layers_5(x_5_0)+x_5_res
   
        x_full=torch.concatenate([x_3,x_5],dim=1)

        # 展平卷积输出，得到适合全连接层的形状
        x_full = x_full.view(x_full.size(0), -1)  # 展平
        
        # 全连接层部分
        out = self.fc_layers(x_full)  # 通过全连接层映射回输入维度
        out_res=self.res(out)
        out=out.reshape([out.size(0),64,8])
        
        final,_=self.lstm(out)
        final=final.reshape([out.size(0),512])
        #final=final+out_res


        return final

    
#代办：做两次次卷积作为16个tokens试试看
#先卷积出16个
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
    
'''class ConvAttentionModel(nn.Module):
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
        
        self.conv1d5=nn.Sequential(        
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2)
            )
        
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

        self.dropout = nn.Dropout(0.5)
        # 自注意力模块
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=32, batch_first=True,dropout=0.2)
        # 全连接部分
        self.linear = nn.Sequential(
            nn.Linear(128*32*3, 64*32),  # 通过全连接层将特征图映射到输入大小
            nn.ReLU(),
            nn.Linear(64*32, 64*16),  # 通过全连接层将特征图映射到输入大小
            nn.ReLU(),            
            nn.Linear(64*16, 512)     
        )
    def forward(self, x):
        batch_size=x.size(0)
        x = x.view(batch_size, 1, -1) 
        x_1=self.conv1d3(x)
        x_2=self.conv1d5(x)
        x_3=self.conv1d7(x)

        #x=self.dropout(x)
        x=torch.concatenate([x_1,x_2,x_3],dim=1)
        #x=self.dropout(x)
        # Step 3: 注意力
        output, attn_weights = self.attention(x, x, x)  # (batch_size, num_segments, embed_dim)



        # Step 4: 展平输出
        output = output.reshape(batch_size, 128*32*3)  # (batch_size, seq_len)
        output=self.linear(output)

        return output'''
    

class ConvAttentionModel_2(nn.Module):
    def __init__(self):
        super(ConvAttentionModel_2, self).__init__()

        # 卷积层：提取特征
        self.conv1d1_3 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.conv1d2_3 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        self.conv1d1_5 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, padding=2)
        self.conv1d2_5 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, padding=2)

        self.conv1d1_7 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
        self.conv1d2_7 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3)

    

        self.avg=nn.AvgPool1d(kernel_size=2, stride=2) 
        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        # 自注意力模块
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True,dropout=0.3)
        # 全连接部分
        self.linear = nn.Sequential(

            nn.Linear(64*64, 64*16),  # 通过全连接层将特征图映射到输入大小
            nn.ReLU(),            
            nn.Linear(64*16, 512)   
        )
    def forward(self, x):
        batch_size=x.size(0)
        x = x.view(batch_size, 1, -1) 
        x=self.conv1d1_3(x)
        x=self.relu(x)


        x=self.conv1d2_3(x)
        x=self.relu(x)
        x=self.avg(x)

        x=self.conv1d1_5(x)
        x=self.relu(x)
        
        x=self.conv1d2_5(x)
        x=self.relu(x)
        x=self.avg(x)

        x=self.conv1d1_7(x)
        x=self.relu(x)

        x=self.conv1d2_7(x)
        x=self.relu(x)
        x=self.avg(x)

        #x=self.dropout(x)

        # Step 3: 注意力
        output, attn_weights = self.attention(x, x, x)  # (batch_size, num_segments, embed_dim)



        # Step 4: 展平输出
        output = output.reshape(batch_size, 64*64)  # (batch_size, seq_len)
        output=self.linear(output)

        return output
    



class SegmentAttentionModel(nn.Module):
    def __init__(self, num_segments=8, embed_dim=64, num_heads=8, kernel_size=3):
        super(SegmentAttentionModel, self).__init__()
        self.num_segments = num_segments
        self.embed_dim = embed_dim

        # 卷积层：提取特征
        self.conv1d = nn.Conv1d(in_channels=num_segments, out_channels=4* num_segments, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv1d2 = nn.Conv1d(in_channels=4*num_segments, out_channels=16* num_segments, kernel_size=kernel_size+4, padding=kernel_size // 2+2)
        self.conv1d_back = nn.Conv1d(in_channels=4*num_segments, out_channels=2*num_segments, kernel_size=kernel_size+4, padding=kernel_size // 2+2)
        self.conv1d_back2 = nn.Conv1d(in_channels=2*num_segments, out_channels=num_segments, kernel_size=kernel_size, padding=kernel_size // 2)

        self.conv_embed1=nn.Conv1d(in_channels=embed_dim, out_channels=4* embed_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv_embed2=nn.Conv1d(in_channels=4*embed_dim, out_channels=16* embed_dim, kernel_size=kernel_size+4, padding=kernel_size // 2+2)
        self.conv_embed2_back=nn.Conv1d(in_channels=4*embed_dim, out_channels=2* embed_dim, kernel_size=kernel_size+4, padding=kernel_size // 2+2)
        self.conv_embed1_back=nn.Conv1d(in_channels=2*embed_dim, out_channels= embed_dim, kernel_size=kernel_size, padding=kernel_size // 2)

        self.avg1=nn.AvgPool1d(kernel_size=2, stride=2) 
        self.avg2=nn.AvgPool1d(kernel_size=2, stride=2) 
        self.relu1=nn.ReLU()
        self.relu2=nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        # 自注意力模块
        '''self.bn1=nn.BatchNorm1d(4 * num_segments)
        self.bn2=nn.BatchNorm1d(8*num_segments)
        self.bn_em1=nn.BatchNorm1d(4 * embed_dim)
        self.bn_em2=nn.BatchNorm1d(8 * embed_dim)'''
        self.attention = nn.MultiheadAttention(embed_dim=int(embed_dim/4), num_heads=num_heads, batch_first=True,dropout=0.5)
        self.linear=nn.Linear(self.embed_dim*self.num_segments*4,self.embed_dim*self.num_segments)
    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size = x.shape[0]

        # Step 1: 切段
        x = x.reshape(batch_size, self.num_segments, self.embed_dim)  # (batch_size, num_segments, embed_dim)
        #x = x.permute(0, 2, 1)
        x=self.conv1d(x)
        x=self.relu1(x)
        x=self.avg1(x)
        #x=self.bn_em1(x)
        x=self.conv1d2(x)
        x=self.relu2(x)
        x=self.avg2(x)
        #x=self.bn_em2(x)
        #x=x.permute(0,2,1)
        #x=self.dropout(x)


        # Step 2: 卷积提取特征
        #x = self.conv1d(x)  # (batch_size, 4*num_segments, embed_dim)
        #x=self.bn1(x)
        #x= self.conv1d2(x)
        #self.bn2(x)
        # Step 3: 注意力
        attn_out, attn_weights = self.attention(x, x, x)  # (batch_size, num_segments, embed_dim)

        #
        #output=self.conv1d_back(attn_out)
        #output=self.conv1d_back2(output)
        output=attn_out
        #output= output.permute(0, 2, 1)
        #output=self.conv_embed2_back(output)
        #output=self.conv_embed1_back(output)
        #output= output.permute(0, 2, 1)

        # Step 4: 展平输出
        output = output.reshape(batch_size, self.embed_dim*self.num_segments*4)  # (batch_size, seq_len)
        output=self.linear(output)

        return output
    


class SegmentAttentionModel_2(nn.Module):
    def __init__(self, num_segments=8, embed_dim=64, num_heads=8, kernel_size=3):
        super(SegmentAttentionModel_2, self).__init__()
        self.num_segments = num_segments
        self.embed_dim = embed_dim

        # 卷积层：提取特征
        self.conv1d = nn.Conv1d(in_channels=num_segments, out_channels=2* num_segments, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv1d2 = nn.Conv1d(in_channels=2*num_segments, out_channels=4* num_segments, kernel_size=kernel_size+4, padding=kernel_size // 2+2)
        self.conv1d_back = nn.Conv1d(in_channels=4*num_segments, out_channels=2*num_segments, kernel_size=kernel_size+4, padding=kernel_size // 2+2)
        self.conv1d_back2 = nn.Conv1d(in_channels=2*num_segments, out_channels=num_segments, kernel_size=kernel_size, padding=kernel_size // 2)

        self.conv_embed1=nn.Conv1d(in_channels=embed_dim, out_channels=2* embed_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv_embed2=nn.Conv1d(in_channels=2*embed_dim, out_channels=4* embed_dim, kernel_size=kernel_size+4, padding=kernel_size // 2+2)
        self.conv_embed2_back=nn.Conv1d(in_channels=4*embed_dim, out_channels=2* embed_dim, kernel_size=kernel_size+4, padding=kernel_size // 2+2)
        self.conv_embed1_back=nn.Conv1d(in_channels=2*embed_dim, out_channels= embed_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        # 自注意力模块
        '''self.bn1=nn.BatchNorm1d(4 * num_segments)
        self.bn2=nn.BatchNorm1d(8*num_segments)
        self.bn_em1=nn.BatchNorm1d(4 * embed_dim)
        self.bn_em2=nn.BatchNorm1d(8 * embed_dim)'''
        self.attention = nn.MultiheadAttention(embed_dim=4*embed_dim, num_heads=num_heads, batch_first=True,dropout=0.3)
        self.linear=nn.Linear(self.embed_dim*self.num_segments,int(self.embed_dim*self.num_segments/2))
    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size = x.shape[0]

        # Step 1: 切段
        x = x.reshape(batch_size, self.num_segments, self.embed_dim)  # (batch_size, num_segments, embed_dim)
        x = x.permute(0, 2, 1)
        x=self.conv_embed1(x)
        #x=self.bn_em1(x)
        x=self.conv_embed2(x)
        #x=self.bn_em2(x)
        x=x.permute(0,2,1)


        # Step 2: 卷积提取特征
        #x = self.conv1d(x)  # (batch_size, 4*num_segments, embed_dim)
        #x=self.bn1(x)
        #x= self.conv1d2(x)
        #self.bn2(x)
        # Step 3: 注意力
        attn_out, attn_weights = self.attention(x, x, x)  # (batch_size, num_segments, embed_dim)

        #
        #output=self.conv1d_back(attn_out)
        #output=self.conv1d_back2(output)
        output=attn_out
        output= output.permute(0, 2, 1)
        output=self.conv_embed2_back(output)
        output=self.conv_embed1_back(output)
        output= output.permute(0, 2, 1)

        # Step 4: 展平输出
        output = output.reshape(batch_size, self.embed_dim*self.num_segments)  # (batch_size, seq_len)
        output=self.linear(output)

        return output

    
# 主模型：联合时域和频域模块
class TimeFrequencyModel(nn.Module):
    def __init__(self, num_segments, embed_dim, num_heads,time_input_size=512, freq_hidden_size=1024, tm="self"):
        super(TimeFrequencyModel, self).__init__()
        if tm=="self":
            self.time_module = SegmentAttentionModel(num_segments, embed_dim, num_heads, kernel_size=3)
        self.abs_module =CNNModel(time_input_size)
        self.angle_module = CNNModel(time_input_size)

    def forward(self, x):
        # 时域模块
        time_output = self.time_module(x)

        # 频域模块
        fft_time_output = torch.fft.fft(time_output, dim=-1)  # 时域预测的傅里叶变换
        
        # 频谱的角度和幅值
        angle_time = torch.angle(fft_time_output)
        abs_time = torch.abs(fft_time_output)
        
        # 学习角度信息
        angle_pred = self.angle_module(angle_time)
        abs_pred=self.abs_module(abs_time)
        # 使用幅值和学习到的角度重构频谱
        freq_reconstructed = abs_pred * torch.exp(1j * angle_pred)
        
        # 逆傅里叶变换重建时域信号
        reconstructed_signal = torch.fft.ifft(freq_reconstructed, dim=-1).real
        
        return time_output, abs_pred, angle_pred,reconstructed_signal
    

  
class FrequencyTimeModel(nn.Module):
    def __init__(self, num_segments, embed_dim, num_heads,time_input_size=512,tm="self"):
        super(FrequencyTimeModel, self).__init__()
        if tm=="self":
            self.time_module = SegmentAttentionModel(num_segments, embed_dim, num_heads, kernel_size=3)
        self.abs_module =LittleCNNModel(time_input_size)
        self.angle_module = LittleCNNModel(time_input_size)

    def forward(self, x):
        # 频域模块
        fft_time_output = torch.fft.fft(x, dim=-1)  # 时域预测的傅里叶变换
        
        # 频谱的角度和幅值
        angle_time = torch.angle(fft_time_output)
        abs_time = torch.abs(fft_time_output)
        
        # 学习角度信息
        angle_pred = self.angle_module(angle_time)
        abs_pred=self.abs_module(abs_time)
        # 使用幅值和学习到的角度重构频谱
        freq_reconstructed = abs_pred * torch.exp(1j * angle_pred)
        
        # 逆傅里叶变换重建时域信号
        time_output = torch.abs(torch.fft.ifft(freq_reconstructed, dim=-1))

        # 时域模块
        reconstructed_signal = self.time_module(time_output)
        
        return time_output, abs_pred, angle_pred,reconstructed_signal

class Freq_module(nn.Module):
    def __init__(self):
        super(Freq_module, self).__init__()
        self.freq_module =ComplexCNNModel()
        self.res=lambda x:x
        self.fc_layers=nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,512))
    def forward(self,x):
        # 频域模块
        fft_time_output = torch.fft.fft(x, dim=-1)  # 时域预测的傅里叶变换

        # 学习角度信息
        freq_pred = self.freq_module(fft_time_output)
        freq_res=self.res(fft_time_output)
        # 逆傅里叶变换重建时域信号
        ifft = torch.abs(torch.fft.ifft(freq_pred+freq_res, dim=-1))

        time_output=torch.concatenate([x,ifft],dim=1)

        # 时域模块
        reconstructed_signal = self.fc_layers(time_output)
        return freq_pred,reconstructed_signal
        
class Freq_module_3(nn.Module):
    def __init__(self):
        super(Freq_module_3, self).__init__()
        self.angle_module =LittleCNNModel()
        self.amp_module=LittleCNNModel()
        self.res=lambda x:x
        self.fc_layers=nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,512))
        self.fc_layer1=nn.Linear(512,512)
        self.weight_adress_angle=nn.Linear(512,512)
        self.weight_adress_amp=nn.Linear(512,512)
    def forward(self, x):
        # 频域模块
        fft_time_output = torch.fft.fft(x, dim=-1)  # 时域预测的傅里叶变换

        # 提取前一半数据（正频率部分）
        phase = torch.angle(fft_time_output)  # Phase of x
        amp=torch.abs(fft_time_output)
        phase_masked=torch.sigmoid(self.weight_adress_angle(phase))
        amp_masked=torch.sigmoid(self.weight_adress_angle(phase))
        phase_pred=self.angle_module(phase_masked)
        amp_pred=self.amp_module(amp_masked)
        #phase_res=self.res(phase_pred)
        #amp_res=self.res(amp_pred)
        freq_pred=amp_pred * torch.exp(1j * phase_pred)

        

        # 学习角度信息
        #freq_pred = self.freq_module(fft_time_output)

        # 复制到另一半（负频率部分）
        #freq_pred_full = torch.cat([freq_pred, torch.flip(freq_pred.conj(), dims=[-1])], dim=-1)

        # 频域残差
        #freq_res = self.res(fft_time_output)

        # 逆傅里叶变换重建时域信号
        ifft = torch.fft.ifft(freq_pred, dim=-1)
        ifft=torch.abs(ifft)
        out=self.fc_layer1(ifft)
        #res=self.res(x)
        #ifft=ifft+res
        # 拼接原始信号和重建信号
        time_output = torch.cat([x, out], dim=1)

        # 时域模块
        reconstructed_signal = self.fc_layers(time_output)

        return out, reconstructed_signal


class TF_model(nn.Module):
    def __init__(self):
        super(TF_model, self).__init__()
        self.time_module = ConvAttentionModel()
        self.freq_module =Freq_module()
        

    def forward(self, x):
        time_output=self.time_module(x)
        # 学习角度信息
        freq_pred,reconstructed_signal = self.freq_module(time_output)
        
        return time_output, freq_pred,reconstructed_signal
    
class TF_model_3(nn.Module):
    def __init__(self):
        super(TF_model_3, self).__init__()
        self.time_module = ConvAttentionModel()
        self.freq_module =Freq_module_3()
        

    def forward(self, x):
        time_output=self.time_module(x)
        # 学习角度信息
        freq_pred,reconstructed_signal = self.freq_module(time_output)
        
        return time_output, freq_pred,reconstructed_signal

    

class FrequencyTimeModel_2(nn.Module):
    def __init__(self, num_segments, embed_dim, num_heads,time_input_size=512,tm="self"):
        super(FrequencyTimeModel_2, self).__init__()
        if tm=="self":
            self.time_module = SegmentAttentionModel_2(num_segments, embed_dim*2, num_heads, kernel_size=3)
        self.freq_module =ComplexCNNModel(time_input_size)

    def forward(self, x):
        # 频域模块
        fft_time_output = torch.fft.fft(x, dim=-1)  # 时域预测的傅里叶变换

        # 学习角度信息
        freq_pred = self.freq_module(fft_time_output)

        # 逆傅里叶变换重建时域信号
        time_output = torch.abs(torch.fft.ifft(freq_pred, dim=-1))
        mixed=torch.concatenate([x,time_output],dim=-1)
        # 时域模块
        reconstructed_signal = self.time_module(mixed)
        
        return time_output, freq_pred,reconstructed_signal
    
class double_branch_v1(nn.Module):
    def __init__(self,num_segments, embed_dim, num_heads):
        super(double_branch_v1, self).__init__()
        self.att=SegmentAttentionModel(num_segments, embed_dim, num_heads)
        self.fc_mix=nn.Linear(1024, 512)


    def forward(self, x1,x2):
        #x1是经过频域处理的，x2为原版
        #将x2进行处理后，加入x1进行学习
        x2=self.att(x2)
        mixed=torch.concatenate([x1,x2],dim=-1)
        # 时域模块
        output = self.fc_mix(mixed)
        
        return output
    
class double_branch_v2(nn.Module):
    def __init__(self,num_segments, embed_dim, num_heads):
        super(double_branch_v2, self).__init__()
        self.att1=SegmentAttentionModel(num_segments, embed_dim, num_heads)
        self.att2=SegmentAttentionModel(num_segments, embed_dim, num_heads)
        self.fc_mix=nn.Linear(1024, 512)


    def forward(self, x1,x2):
        x1=self.att1(x1)
        x2=self.att2(x2)
        mixed=torch.concatenate([x1,x2],dim=-1)
        # 时域模块
        output = self.fc_mix(mixed)
        
        return output
    
class FrequencyTimeModel_3(nn.Module):
    def __init__(self, num_segments, embed_dim, num_heads,time_input_size=512):
        super(FrequencyTimeModel_3, self).__init__()
        self.time_module=double_branch_v1(num_segments, embed_dim, num_heads)
        self.freq_module =ComplexCNNModel(time_input_size)

    def forward(self, x):
        # 频域模块
        fft_time_output = torch.fft.fft(x, dim=-1)  # 时域预测的傅里叶变换

        # 学习角度信息
        freq_pred = self.freq_module(fft_time_output)

        # 逆傅里叶变换重建时域信号
        time_output = torch.abs(torch.fft.ifft(freq_pred, dim=-1))
        # 时域模块
        reconstructed_signal = self.time_module(time_output,x)
        
        return time_output, freq_pred,reconstructed_signal
    
class FrequencyTimeModel_4(nn.Module):
    def __init__(self, num_segments, embed_dim, num_heads,time_input_size=512):
        super(FrequencyTimeModel_4, self).__init__()
        self.time_module=double_branch_v2(num_segments, embed_dim, num_heads)
        self.freq_module =ComplexCNNModel(time_input_size)

    def forward(self, x):
        # 频域模块
        fft_time_output = torch.fft.fft(x, dim=-1)  # 时域预测的傅里叶变换

        # 学习角度信息
        freq_pred = self.freq_module(fft_time_output)

        # 逆傅里叶变换重建时域信号
        time_output = torch.abs(torch.fft.ifft(freq_pred, dim=-1))
        # 时域模块
        reconstructed_signal = self.time_module(time_output,x)
        
        return time_output, freq_pred,reconstructed_signal