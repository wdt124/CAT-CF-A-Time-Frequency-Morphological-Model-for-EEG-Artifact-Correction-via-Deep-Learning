from train_arch import *

def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True




# 训练函数
def train_model_step(model, train_loader, val_loader, epochs_freq=5,epochs_time=5, lr=0.000001, device='cpu', outdir='Models/1119_05'):
    criterion = nn.MSELoss().to(device)
    freq_criterion=ComplexMSELoss().to(device)
    optimizer_time = optim.Adam(model.time_module.parameters(), lr=lr)
    scheduler_time = StepLR(optimizer_time, step_size=10, gamma=0.9995)
    optimizer_freq = optim.Adam(model.freq_module.parameters(), lr=lr)
    scheduler_freq = StepLR(optimizer_freq, step_size=10, gamma=0.9995)
    # 确保输出目录存在
    os.makedirs(outdir, exist_ok=True)
    # 创建模型保存目录
    model_save_dir = os.path.join(outdir, 'model_saved')
    os.makedirs(model_save_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_model_state = None
    time_output=[]
    outputs=[]
    X_batch_for_freq=[]
    # 用于保存 loss
    train_losses_freq = []
    val_losses_freq = []
    train_losses_time = []
    val_losses_time = []

    freeze_module(model.time_module)
    for epoch in range(epochs_freq):
        # 训练频域阶段
        model.train()
        train_loss_freq = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer_freq.zero_grad()
            time_output,freq_pred,outputs = model(X_batch)
            loss=freq_criterion(freq_pred, y_batch) 
            loss.backward()
            optimizer_freq.step()
            train_loss_freq += loss.item()
            X_batch_for_freq=X_batch
        # 更新学习率
        scheduler_freq.step()
        # 验证阶段
        model.eval()
        val_loss_freq = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                time_output,freq_pred,outputs = model(X_batch)
                loss=freq_criterion(freq_pred, y_batch) 
                val_loss_freq += loss.item()

        # 保存每 epoch 的 loss
        train_losses_freq.append(train_loss_freq / len(train_loader))
        val_losses_freq.append(val_loss_freq / len(val_loader))

        # 记录最佳模型
        if val_loss_freq < best_val_loss:
            best_val_loss = val_loss_freq
            best_model_state = model.state_dict()

        # 每隔 5 个 epoch 保存一次模型
        if (epoch + 1) % 5 == 0:
            model_save_path = f'{outdir}/model_saved/freq_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

        # 打印当前 epoch 的 loss
        print(f"Epoch {epoch+1}/{epochs_freq}, Train Loss of freq: {train_loss_freq/len(train_loader):.4f}, Val Loss of freq: {val_loss_freq/len(val_loader):.4f}")
    np.save(f'{outdir}/model_saved/time_output.npy',time_output.cpu().numpy())
    np.save(f'{outdir}/model_saved/X.npy',X_batch_for_freq.cpu().numpy())
    np.save(f'{outdir}/model_saved/freq_pred.npy',freq_pred.cpu().numpy())

    unfreeze_module(model.time_module)
    freeze_module(model.freq_module)
    for epoch in range(epochs_time):
        # 训练时域阶段
        model.train()
        train_loss_time = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer_time.zero_grad()
            time_output,freq_pred,outputs = model(X_batch)
            loss=criterion(outputs, y_batch)
            loss.backward()
            optimizer_time.step()
            train_loss_time += loss.item()
        # 更新学习率
        scheduler_time.step()
        # 验证阶段
        model.eval()
        val_loss_time = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                time_output,freq_pred,outputs = model(X_batch)
                loss=criterion(outputs, y_batch)
                val_loss_time += loss.item()

        # 保存每 epoch 的 loss
        train_losses_time.append(train_loss_time / len(train_loader))
        val_losses_time.append(val_loss_time / len(val_loader))

        # 记录最佳模型
        if val_loss_time < best_val_loss:
            best_val_loss = val_loss_time
            best_model_state = model.state_dict()

        # 每隔 5 个 epoch 保存一次模型
        if (epoch + 1) % 5 == 0:
            model_save_path = f'{outdir}/model_saved/time_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

        # 打印当前 epoch 的 loss
        print(f"Epoch {epoch+1}/{epochs_time}, Train Loss of time: {train_loss_time/len(train_loader):.4f}, Val Loss: {val_loss_time/len(val_loader):.4f}")


    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 保存最终 loss 曲线
    plot_loss_curve(train_losses_freq, val_losses_freq, outdir,title='Loss of Freq_Module')
    plot_loss_curve(train_losses_time, val_losses_time, outdir,title='Loss of Time_Module')


# 训练函数
def train_TF_step(model, train_loader, val_loader, epochs_freq=5,epochs_time=5,epochs_fusion=5, lr_freq=0.000001,lr_time=0.000001,lr_fu=0.0001, device='cpu', outdir='Models/1119_05',time_path=None,freq_path=None,return_loss=False):
    freq_loss_path=outdir+'/freq_loss'
    time_loss_path=outdir+'/time_loss'
    fusion_loss_path=outdir+'/fusion_loss'
    criterion = nn.MSELoss().to(device)
    freq_criterion=ComplexMSELoss().to(device)
    optimizer_time = optim.Adam(model.time_module.parameters(), lr=lr_time)
    scheduler_time = StepLR(optimizer_time, step_size=20, gamma=0.9995)
    optimizer_freq = optim.Adam(model.freq_module.parameters(), lr=lr_freq)
    scheduler_freq = StepLR(optimizer_freq, step_size=20, gamma=0.9995)
    optimizer_fusion = optim.Adam(model.fusion_module.parameters(), lr=lr_fu)
    scheduler_fusion = StepLR(optimizer_fusion, step_size=20, gamma=0.9995)
    # 确保输出目录存在
    os.makedirs(outdir, exist_ok=True)
    # 创建模型保存目录
    model_save_dir = os.path.join(outdir, 'model_saved')
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(freq_loss_path, exist_ok=True)
    os.makedirs(time_loss_path, exist_ok=True)
    os.makedirs(fusion_loss_path, exist_ok=True)
    best_val_loss_freq = float('inf')
    best_val_loss_time = float('inf')
    best_val_loss_fusion = float('inf')
    best_model_state = None
    time_output=[]
    outputs=[]
    X_batch_for_freq=[]
    # 用于保存 loss
    train_losses_freq = []
    val_losses_freq = []
    train_losses_time = []
    val_losses_time = []
    train_losses_fusion = []
    val_losses_fusion = []
    freeze_module(model.freq_module)
    freeze_module(model.fusion_module)
    if time_path==None:
        for epoch in range(epochs_time):
            # 训练时域阶段
            model.train()
            train_loss_time = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer_time.zero_grad()
                time_output,_,_ = model(X_batch)
                loss=criterion(time_output, y_batch) 
                loss.backward()
                optimizer_time.step()
                train_loss_time += loss.item()
                X_batch_for_time=X_batch
            # 更新学习率
            scheduler_time.step()
            # 验证阶段
            model.eval()
            val_loss_time = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    time_output,_,_ = model(X_batch)
                    loss=criterion(time_output, y_batch) 
                    val_loss_time += loss.item()

            # 保存每 epoch 的 loss
            train_losses_time.append(train_loss_time / len(train_loader))
            val_losses_time.append(val_loss_time / len(val_loader))

            # 记录最佳模型
            if val_loss_time < best_val_loss_time:
                best_val_loss_time = val_loss_time
                best_model_state = model.state_dict()

            # 每隔 5 个 epoch 保存一次模型
            if (epoch + 1) % 50 == 0:
                model_save_path = f'{outdir}/model_saved/time_model_epoch_{epoch+1}.pth'
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at {model_save_path}")

            # 打印当前 epoch 的 loss
            print(f"Epoch {epoch+1}/{epochs_time}, Train Loss of time: {train_loss_time/len(train_loader):.4f}, Val Loss of time: {val_loss_time/len(val_loader):.4f}")
        '''np.save(f'{outdir}/model_saved/time_output.npy',time_output.cpu().numpy())
        np.save(f'{outdir}/model_saved/X.npy',X_batch_for_freq.cpu().numpy())
        np.save(f'{outdir}/model_saved/freq_pred.npy',freq_pred.cpu().numpy())'''
    else:
        # 加载已经训练好的A模型参数
        model.time_module.load_state_dict(torch.load(time_path))

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    unfreeze_module(model.freq_module)
    freeze_module(model.time_module)

    if not freq_path==None:
        model.freq_module.freq_module.load_state_dict(torch.load(freq_path))
        freeze_module(model.freq_module.freq_module)
    for epoch in range(epochs_freq):
        # 训练频域阶段
        model.train()
        train_loss_freq = 0.0
        #_loss=0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer_freq.zero_grad()
            _,freq_pred,outputs = model(X_batch)
            loss_f=freq_criterion(freq_pred, y_batch)
            #loss=criterion(outputs, y_batch)
            loss_f.backward()
            optimizer_freq.step()
            #train_loss_freq += loss.item()
            train_loss_freq+=loss_f.item()
        # 更新学习率
        scheduler_freq.step()
        # 验证阶段
        model.eval()
        val_loss_freq = 0.0
        #f_val=0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                time_output,freq_pred,outputs = model(X_batch)
                #loss=criterion(outputs, y_batch)
                f_val_loss=freq_criterion(freq_pred, y_batch)
                #val_loss_freq += loss.item()
                val_loss_freq+=f_val_loss

        # 保存每 epoch 的 loss
        train_losses_freq.append(train_loss_freq / len(train_loader))
        val_losses_freq.append(val_loss_freq / len(val_loader))

        # 记录最佳模型
        if val_loss_freq < best_val_loss_freq:
            best_val_loss_freq = val_loss_freq
            best_model_state = model.state_dict()

        # 每隔 5 个 epoch 保存一次模型
        if (epoch + 1) % 50 == 0:
            model_save_path = f'{outdir}/model_saved/freq_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

        # 打印当前 epoch 的 loss
        print(f"Epoch {epoch+1}/{epochs_freq}, Train Loss of freq: {train_loss_freq/len(train_loader):.4f}, Val Loss of freq: {val_loss_freq/len(val_loader):.4f}")

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    freeze_module(model.freq_module)
    unfreeze_module(model.fusion_module)

    for epoch in range(epochs_fusion):
        # 训练频域阶段
        model.train()
        train_loss_fusion = 0.0
        #_loss=0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer_fusion.zero_grad()
            _,_,outputs = model(X_batch)
            loss_f=criterion(outputs, y_batch)
            #loss=criterion(outputs, y_batch)
            loss_f.backward()
            optimizer_fusion.step()
            #train_loss_freq += loss.item()
            train_loss_fusion += loss_f.item()
        # 更新学习率
        scheduler_fusion.step()
        # 验证阶段
        model.eval()
        val_loss_fusion = 0.0
        #f_val=0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                _, _, outputs = model(X_batch)
                #loss=criterion(outputs, y_batch)
                fusion_loss=criterion(outputs, y_batch)
                #val_loss_freq += loss.item()
                val_loss_fusion += fusion_loss.item()

        # 保存每 epoch 的 loss
        train_losses_fusion.append(train_loss_fusion / len(train_loader))
        val_losses_fusion.append(val_loss_fusion / len(val_loader))

        # 记录最佳模型
        if val_loss_fusion < best_val_loss_fusion:
            best_val_loss_fusion = val_loss_fusion
            best_model_state = model.state_dict()


        # 每隔 10 个 epoch 保存一次模型
        if (epoch + 1) % 50 == 0:
            model_save_path = f'{outdir}/model_saved/fusion_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

        # 打印当前 epoch 的 loss
        print(f"Epoch {epoch+1}/{epochs_fusion}, Train Loss of fusion: {train_loss_fusion/len(train_loader):.4f}, Val Loss of fusion: {val_loss_fusion/len(val_loader):.4f}")


    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 保存最终 loss 曲线
    plot_loss_curve(train_losses_freq, val_losses_freq, freq_loss_path,title='Loss of Freq_Module')
    plot_loss_curve(train_losses_time, val_losses_time,time_loss_path,title='Loss of Time_Module')
    plot_loss_curve(train_losses_fusion, val_losses_fusion, fusion_loss_path,title='Loss of Fusion_Module')
    if return_loss:
        train_losses_freq = [float(l.cpu()) if torch.is_tensor(l) else float(l) for l in train_losses_freq]
        val_losses_freq = [float(l.cpu()) if torch.is_tensor(l) else float(l) for l in val_losses_freq]
        
        return train_losses_freq, val_losses_freq, train_losses_time, val_losses_time,train_losses_fusion, val_losses_fusion


# 训练函数
def train_model_step_in_epoch(model, train_loader, val_loader, epochs=5, lr_time=0.000001,lr_freq=0.000001, device='cpu', outdir='Models/1119_05'):
    criterion = nn.MSELoss().to(device)
    freq_criterion=ComplexMSELoss().to(device)
    optimizer_time = optim.Adam(model.time_module.parameters(), lr=lr_time)
    scheduler_time = StepLR(optimizer_time, step_size=10, gamma=0.9995)
    optimizer_freq = optim.Adam(model.freq_module.parameters(), lr=lr_freq)
    scheduler_freq = StepLR(optimizer_freq, step_size=10, gamma=0.9995)
    # 确保输出目录存在
    os.makedirs(outdir, exist_ok=True)
    # 创建模型保存目录
    model_save_dir = os.path.join(outdir, 'model_saved')
    os.makedirs(model_save_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_model_state = None

    # 用于保存 loss
    train_losses_freq = []
    val_losses_freq = []
    train_losses_time = []
    val_losses_time = []

    
    for epoch in range(epochs):
        freeze_module(model.time_module)
        # 训练频域阶段
        model.train()
        train_loss_freq = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer_freq.zero_grad()
            time_output,freq_pred,outputs = model(X_batch)
            loss=freq_criterion(freq_pred, y_batch) 
            loss.backward()
            optimizer_freq.step()
            train_loss_freq += loss.item()
        # 更新学习率
        scheduler_freq.step()
        # 验证阶段
        model.eval()
        val_loss_freq = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                time_output,freq_pred,outputs = model(X_batch)
                loss=freq_criterion(freq_pred, y_batch) 
                val_loss_freq += loss.item()

        # 保存每 epoch 的 loss
        train_losses_freq.append(train_loss_freq / len(train_loader))
        val_losses_freq.append(val_loss_freq / len(val_loader))

        # 记录最佳模型
        if val_loss_freq < best_val_loss:
            best_val_loss = val_loss_freq
            best_model_state = model.state_dict()

        # 每隔 5 个 epoch 保存一次模型
        if (epoch + 1) % 5 == 0:
            model_save_path = f'{outdir}/model_saved/freq_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

        # 打印当前 epoch 的 loss
        print(f"Epoch {epoch+1}/{epochs}, Train Loss of freq: {train_loss_freq/len(train_loader):.4f}, Val Loss of freq: {val_loss_freq/len(val_loader):.4f}")
    
        unfreeze_module(model.time_module)
        freeze_module(model.freq_module)
        # 训练时域阶段
        model.train()
        train_loss_time = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer_time.zero_grad()
            time_output,freq_pred,outputs = model(X_batch)
            loss=criterion(outputs, y_batch)
            loss.backward()
            optimizer_time.step()
            train_loss_time += loss.item()
        # 更新学习率
        scheduler_time.step()
        # 验证阶段
        model.eval()
        val_loss_time = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                time_output,freq_pred,outputs = model(X_batch)
                loss=criterion(outputs, y_batch)
                val_loss_time += loss.item()

        # 保存每 epoch 的 loss
        train_losses_time.append(train_loss_time / len(train_loader))
        val_losses_time.append(val_loss_time / len(val_loader))

        # 记录最佳模型
        if val_loss_time < best_val_loss:
            best_val_loss = val_loss_time
            best_model_state = model.state_dict()
        unfreeze_module(model.freq_module)
        # 每隔 5 个 epoch 保存一次模型
        if (epoch + 1) % 5 == 0:
            model_save_path = f'{outdir}/model_saved/time_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

        # 打印当前 epoch 的 loss
        print(f"Epoch {epoch+1}/{epochs}, Train Loss of time: {train_loss_time/len(train_loader):.4f}, Val Loss: {val_loss_time/len(val_loader):.4f}")


    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 保存最终 loss 曲线
    plot_loss_curve(train_losses_freq, val_losses_freq, outdir,title='Loss of Freq_Module')
    plot_loss_curve(train_losses_time, val_losses_time, outdir,title='Loss of Time_Module')


# 训练函数
def train_tf_step_in_epoch(model, train_loader, val_loader, epochs=5, lr_time=0.000001,lr_freq=0.000001, device='cpu', outdir='Models/1119_05'):
    criterion = nn.MSELoss().to(device)
    freq_criterion=ComplexMSELoss().to(device)
    optimizer_time = optim.Adam(model.time_module.parameters(), lr=lr_time)
    scheduler_time = StepLR(optimizer_time, step_size=10, gamma=0.9995)
    optimizer_freq = optim.Adam(model.freq_module.parameters(), lr=lr_freq)
    scheduler_freq = StepLR(optimizer_freq, step_size=10, gamma=0.9995)
    # 确保输出目录存在
    os.makedirs(outdir, exist_ok=True)
    # 创建模型保存目录
    model_save_dir = os.path.join(outdir, 'model_saved')
    os.makedirs(model_save_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_model_state = None

    # 用于保存 loss
    train_losses_freq = []
    val_losses_freq = []
    train_losses_time = []
    val_losses_time = []

    
    for epoch in range(epochs):
        freeze_module(model.time_module)
        # 训练频域阶段
        model.train()
        train_loss_freq = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer_freq.zero_grad()
            time_output,freq_pred,outputs = model(X_batch)
            loss=freq_criterion(freq_pred, y_batch) 
            loss.backward()
            optimizer_freq.step()
            train_loss_freq += loss.item()
        # 更新学习率
        scheduler_freq.step()
        # 验证阶段
        model.eval()
        val_loss_freq = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                time_output,freq_pred,outputs = model(X_batch)
                loss=freq_criterion(freq_pred, y_batch) 
                val_loss_freq += loss.item()

        # 保存每 epoch 的 loss
        train_losses_freq.append(train_loss_freq / len(train_loader))
        val_losses_freq.append(val_loss_freq / len(val_loader))

        # 记录最佳模型
        if val_loss_freq < best_val_loss:
            best_val_loss = val_loss_freq
            best_model_state = model.state_dict()

        # 每隔 5 个 epoch 保存一次模型
        if (epoch + 1) % 5 == 0:
            model_save_path = f'{outdir}/model_saved/freq_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

        # 打印当前 epoch 的 loss
        print(f"Epoch {epoch+1}/{epochs}, Train Loss of freq: {train_loss_freq/len(train_loader):.4f}, Val Loss of freq: {val_loss_freq/len(val_loader):.4f}")
    
        unfreeze_module(model.time_module)
        freeze_module(model.freq_module)
        # 训练时域阶段
        model.train()
        train_loss_time = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer_time.zero_grad()
            time_output,freq_pred,outputs = model(X_batch)
            loss=criterion(outputs, y_batch)
            loss.backward()
            optimizer_time.step()
            train_loss_time += loss.item()
        # 更新学习率
        scheduler_time.step()
        # 验证阶段
        model.eval()
        val_loss_time = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                time_output,freq_pred,outputs = model(X_batch)
                loss=criterion(outputs, y_batch)
                val_loss_time += loss.item()

        # 保存每 epoch 的 loss
        train_losses_time.append(train_loss_time / len(train_loader))
        val_losses_time.append(val_loss_time / len(val_loader))

        # 记录最佳模型
        if val_loss_time < best_val_loss:
            best_val_loss = val_loss_time
            best_model_state = model.state_dict()
        unfreeze_module(model.freq_module)
        # 每隔 5 个 epoch 保存一次模型
        if (epoch + 1) % 5 == 0:
            model_save_path = f'{outdir}/model_saved/time_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

        # 打印当前 epoch 的 loss
        print(f"Epoch {epoch+1}/{epochs}, Train Loss of time: {train_loss_time/len(train_loader):.4f}, Val Loss: {val_loss_time/len(val_loader):.4f}")


    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 保存最终 loss 曲线
    plot_loss_curve(train_losses_freq, val_losses_freq, outdir,title='Loss of Freq_Module')
    plot_loss_curve(train_losses_time, val_losses_time, outdir,title='Loss of Time_Module')


