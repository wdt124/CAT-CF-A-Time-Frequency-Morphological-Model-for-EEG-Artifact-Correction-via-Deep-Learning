import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from loss_arch import *
# 1. 数据集类定义
class MyDataset(Dataset):
    def __init__(self, X, y, device='cpu'):
        """
        初始化数据集
        :param X: 输入特征，形状为 (samples, 512)
        :param y: 输出标签，形状为 (samples, 512)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.X.to(device)
        self.y.to(device)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        返回指定索引的样本和标签
        """
        return self.X[idx], self.y[idx]


# 3. 训练函数
def train_model_vanilla(model, train_loader, val_loader, epochs=5, lr=0.000001,device='cpu', outdir='Models/1119_05', return_loss=False):
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.9995)

    # 确保输出目录存在
    os.makedirs(outdir, exist_ok=True)
    # 创建模型保存目录
    model_save_dir = os.path.join(outdir, 'model_saved')
    os.makedirs(model_save_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_model_state = None

    # 用于保存 loss
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()

            #time_output,abs_pred,angle_pred,outputs = model(X_batch)
            #loss = criterion(outputs, y_batch)+angle_criterion(angle_pred, y_batch,device=device)+abs_criterion(abs_pred,y_batch,device=device)
            outputs = model(X_batch)
            loss=criterion(outputs, y_batch) 

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 更新学习率
        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                #time_output,abs_pred,angle_pred,outputs = model(X_batch)
                #loss = criterion(outputs, y_batch)+angle_criterion(angle_pred, y_batch,device=device)+abs_criterion(abs_pred,y_batch,device=device)
                outputs = model(X_batch)
                loss=criterion(outputs, y_batch) 
                #print(loss_f,loss_t)               

                val_loss += loss.item()

        # 保存每 epoch 的 loss
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        # 记录最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        # 每隔 5 个 epoch 保存一次模型
        if (epoch + 1) % 10 == 0:
            model_save_path = f'{outdir}/model_saved/model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

        # 打印当前 epoch 的 loss
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f},, Val Loss: {val_loss/len(val_loader):.4f}")

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 保存最终 loss 曲线
    plot_loss_curve(train_losses, val_losses, outdir)
    if return_loss:
        return train_losses, val_losses


# 训练函数
def train_model(model, train_loader, val_loader, epochs=5, lr=0.000001, device='cpu', outdir='Models/1119_05',tf='TF'):
    criterion = nn.MSELoss().to(device)
    angle_criterion=PhaseSpectrumPenalty().to(device)
    abs_criterion=AbsSpectrumPenalty().to(device)
    freq_criterion=ComplexMSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.99)

    # 确保输出目录存在
    os.makedirs(outdir, exist_ok=True)
    # 创建模型保存目录
    model_save_dir = os.path.join(outdir, 'model_saved')
    os.makedirs(model_save_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_model_state = None

    # 用于保存 loss
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            if tf=='TF':
                time_output,abs_pred,angle_pred,outputs = model(X_batch)
                loss = criterion(time_output, y_batch)+angle_criterion(angle_pred, y_batch,device=device)+abs_criterion(abs_pred,y_batch,device=device)
            elif tf=='FT':
                #time_output,abs_pred,angle_pred,outputs = model(X_batch)
                #loss = criterion(outputs, y_batch)+angle_criterion(angle_pred, y_batch,device=device)+abs_criterion(abs_pred,y_batch,device=device)
                time_output,freq_pred,outputs = model(X_batch)
                loss_f=freq_criterion(freq_pred, y_batch) 
                loss_t = criterion(outputs, y_batch)  
                loss=loss_f+loss_t  
            else:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)+angle_criterion(outputs, y_batch,device=device)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 更新学习率
        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                if tf=='TF':
                    time_output,abs_pred,angle_pred,outputs = model(X_batch)
                    loss=criterion(time_output, y_batch)+angle_criterion(angle_pred, y_batch,device=device)+abs_criterion(abs_pred,y_batch,device=device)
                elif tf=='FT':
                    #time_output,abs_pred,angle_pred,outputs = model(X_batch)
                    #loss = criterion(outputs, y_batch)+angle_criterion(angle_pred, y_batch,device=device)+abs_criterion(abs_pred,y_batch,device=device)
                    time_output,freq_pred,outputs = model(X_batch)
                    loss_f=freq_criterion(freq_pred, y_batch) 
                    loss_t = criterion(outputs, y_batch)  
                    loss=loss_f+loss_t
                    #print(loss_f,loss_t)               
                else:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)+angle_criterion(outputs, y_batch,device=device)
                val_loss += loss.item()

        # 保存每 epoch 的 loss
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        # 记录最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        # 每隔 5 个 epoch 保存一次模型
        if (epoch + 1) % 5 == 0:
            model_save_path = f'{outdir}/model_saved/model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

        # 打印当前 epoch 的 loss
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f},, Val Loss: {val_loss/len(val_loader):.4f}")

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 保存最终 loss 曲线
    plot_loss_curve(train_losses, val_losses, outdir)






# 绘制并保存损失曲线
def plot_loss_curve(train_losses, val_losses, outdir,title='Training and Validation Loss'):
    train_losses = [float(l.cpu()) if torch.is_tensor(l) else float(l) for l in train_losses]
    val_losses = [float(l.cpu()) if torch.is_tensor(l) else float(l) for l in val_losses]

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # 保存损失曲线
    loss_curve_path = f'{outdir}/loss_curve.png'
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Loss curve saved at {loss_curve_path}")


# 4. 测试函数
def test_model(model, test_loader,outdir='Models/1119_05',device='cpu',tf=False):
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss().to(device)
    angle_criterion=PhaseSpectrumPenalty().to(device)
    abs_criterion=AbsSpectrumPenalty().to(device)
    freq_criterion=ComplexMSELoss().to(device)
    
    # 创建一个数组来保存预测结果和真实标签
    all_predictions = []
    all_true_labels = []
    all_ori=[]

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if tf=='TF':
                time_output,abs_pred,angle_pred,outputs = model(X_batch)
                loss=criterion(time_output, y_batch)+angle_criterion(angle_pred, y_batch,device=device)+abs_criterion(abs_pred,y_batch,device=device)
            elif tf=='FT':
                #time_output,abs_pred,angle_pred,outputs = model(X_batch)
                #loss = criterion(outputs, y_batch)+angle_criterion(angle_pred, y_batch,device=device)+abs_criterion(abs_pred,y_batch,device=device)  
                time_output,freq_pred,outputs = model(X_batch)
                loss_f=freq_criterion(freq_pred, y_batch) 
                loss_t = criterion(outputs, y_batch)  
                loss=loss_f+loss_t                               
            else:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)+angle_criterion(outputs, y_batch,device=device)
            test_loss += loss.item()
            # 保存预测值和真实标签
            all_predictions.append(outputs.cpu().numpy())  # 预测值
            all_true_labels.append(y_batch.cpu().numpy())  # 真实标签
            all_ori.append(X_batch.cpu().numpy())

    # 输出测试损失
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")

    # 保存预测结果和真实标签为 .npy 文件
    np.save(f'{outdir}/predictions.npy', all_predictions)
    np.save(f'{outdir}/true_labels.npy', all_true_labels)
    np.save(f'{outdir}/original_sig.npy', all_ori)


# 4. 测试函数
def test_model_TF(model, test_loader,outdir='Models/1119_05',device='cpu'):
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss().to(device)
    angle_criterion=PhaseSpectrumPenalty().to(device)
    abs_criterion=AbsSpectrumPenalty().to(device)
    freq_criterion=ComplexMSELoss().to(device)
    
    # 创建一个数组来保存预测结果和真实标签
    all_predictions = []
    all_true_labels = []
    all_ori=[]

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            time_output,freq_pred,outputs = model(X_batch)
            loss = criterion(outputs, y_batch)  
                             
            test_loss += loss.item()
            # 保存预测值和真实标签
            all_predictions.append(outputs.cpu().numpy())  # 预测值
            all_true_labels.append(y_batch.cpu().numpy())  # 真实标签
            all_ori.append(X_batch.cpu().numpy())

    # 输出测试损失
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")

    # 保存预测结果和真实标签为 .npy 文件
    np.save(f'{outdir}/predictions.npy', all_predictions)
    np.save(f'{outdir}/true_labels.npy', all_true_labels)
    np.save(f'{outdir}/original_sig.npy', all_ori)


# 4. 测试函数
def test_model_vanilla(model, test_loader,outdir='Models/1119_05',device='cpu'):
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss().to(device)
    
    # 创建一个数组来保存预测结果和真实标签
    all_predictions = []
    all_true_labels = []
    all_ori=[]

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            #time_output,abs_pred,angle_pred,outputs = model(X_batch)
            #loss = criterion(outputs, y_batch)+angle_criterion(angle_pred, y_batch,device=device)+abs_criterion(abs_pred,y_batch,device=device)  
            outputs = model(X_batch)
            loss=criterion(outputs, y_batch) 
                           

            test_loss += loss.item()
            # 保存预测值和真实标签
            all_predictions.append(outputs.cpu().numpy())  # 预测值
            all_true_labels.append(y_batch.cpu().numpy())  # 真实标签
            all_ori.append(X_batch.cpu().numpy())

    # 输出测试损失
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")

    # 保存预测结果和真实标签为 .npy 文件
    np.save(f'{outdir}/predictions.npy', all_predictions)
    np.save(f'{outdir}/true_labels.npy', all_true_labels)
    np.save(f'{outdir}/original_sig.npy', all_ori)




# 4. 测试函数
def test_model_comp(model, test_loader,outdir='Models/1119_05',device='cpu',loss='comp'):
    model.eval()
    test_loss = 0.0
    if loss=='comp':
        criterion = ComplexMSELoss().to(device)
    elif loss=='phase':
        criterion = PhaseSpectrumPenalty().to(device)
    elif loss=='abs':
        criterion = AbsSpectrumPenalty().to(device)
    
    # 创建一个数组来保存预测结果和真实标签
    all_predictions = []
    all_true_labels = []
    all_ori=[]

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            #time_output,abs_pred,angle_pred,outputs = model(X_batch)
            #loss = criterion(outputs, y_batch)+angle_criterion(angle_pred, y_batch,device=device)+abs_criterion(abs_pred,y_batch,device=device)  
            outputs = model(X_batch)
            loss=criterion(outputs, y_batch) 
                           

            test_loss += loss.item()
            # 保存预测值和真实标签
            all_predictions.append(outputs.cpu().numpy())  # 预测值
            all_true_labels.append(y_batch.cpu().numpy())  # 真实标签
            all_ori.append(X_batch.cpu().numpy())

    # 输出测试损失
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")

    # 保存预测结果和真实标签为 .npy 文件
    np.save(f'{outdir}/predictions.npy', all_predictions)
    np.save(f'{outdir}/true_labels.npy', all_true_labels)
    np.save(f'{outdir}/original_sig.npy', all_ori)

# 3. 训练函数
def train_model_comp(model, train_loader, val_loader, epochs=5, lr=0.000001,device='cpu', outdir='Models/1119_05',loss='comp'):
    if loss=='comp':
        criterion = ComplexMSELoss().to(device)
    elif loss=='phase':
        criterion = PhaseSpectrumPenalty().to(device)
    elif loss=='abs':
        criterion = AbsSpectrumPenalty().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.9995)

    # 确保输出目录存在
    os.makedirs(outdir, exist_ok=True)
    # 创建模型保存目录
    model_save_dir = os.path.join(outdir, 'model_saved')
    os.makedirs(model_save_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_model_state = None

    # 用于保存 loss
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()

            #time_output,abs_pred,angle_pred,outputs = model(X_batch)
            #loss = criterion(outputs, y_batch)+angle_criterion(angle_pred, y_batch,device=device)+abs_criterion(abs_pred,y_batch,device=device)
            outputs = model(X_batch)
            loss=criterion(outputs, y_batch) 

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 更新学习率
        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                #time_output,abs_pred,angle_pred,outputs = model(X_batch)
                #loss = criterion(outputs, y_batch)+angle_criterion(angle_pred, y_batch,device=device)+abs_criterion(abs_pred,y_batch,device=device)
                outputs = model(X_batch)
                loss=criterion(outputs, y_batch) 
                #print(loss_f,loss_t)               

                val_loss += loss.item()

        # 保存每 epoch 的 loss
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        # 记录最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        # 每隔 5 个 epoch 保存一次模型
        if (epoch + 1) % 100 == 0:
            model_save_path = f'{outdir}/model_saved/model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

        # 打印当前 epoch 的 loss
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f},, Val Loss: {val_loss/len(val_loader):.4f}")

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 保存最终 loss 曲线
    plot_loss_curve(train_losses, val_losses, outdir)

