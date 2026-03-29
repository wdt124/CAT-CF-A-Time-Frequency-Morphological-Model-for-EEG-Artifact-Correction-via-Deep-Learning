from train_arch import *
from train_arch_step import *
from models_arch import *
from DuoCL_arch import *
from VAE import *
from train_vae import *
from phase_model import *
from regular import *
import multiprocessing
# GPU 设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#思路：循环读取十个数据集的数据，将所保存的模型存储于一个文件夹中
def main(i):
    # 假设你已有数据集 X_train, y_train, X_val, y_val, X_test, y_test
    # 替换为你的实际数据
    outdir=f'F:/CATCF/Models/Snrs_models/snrs_CATCF_r_{i}_new_1110'
    np.random.seed(42)
    X_train = np.load(f'simulation/snr_datasets/g_train_cont_r_{i}_noshuffle.npy')
    y_train = np.load(f'simulation/snr_datasets/g_train_pure_r_{i}_noshuffle.npy')
    X_val = np.load(f'simulation/snr_datasets/g_val_cont_r_{i}_noshuffle.npy')
    y_val = np.load(f'simulation/snr_datasets/g_val_pure_r_{i}_noshuffle.npy')
    X_test = np.load(f'simulation/snr_datasets/g_test_cont_r_{i}_noshuffle.npy')
    y_test = np.load(f'simulation/snr_datasets/g_test_pure_r_{i}_noshuffle.npy')

    # 创建数据加载器
    train_dataset = MyDataset(X_train, y_train,device=device)
    val_dataset = MyDataset(X_val, y_val,device=device)
    test_dataset = MyDataset(X_test, y_test,device=device)


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,drop_last=True)
    # 初始化模型
    model=TF_model_4()
    #model=ConvAttentionModel_2()
    #model=Complex_CNN()
    #model=DuoCL()
    #model=vanilla_RNNModel()
    #model=ComplexCNNModel_deri()
    #model=FCNN()
    print(model)
    model.to(device)
    # 训练模型
    #train_model_vanilla(model, train_loader, val_loader, epochs=200, lr=0.0001,device=device,outdir=outdir)#这是训练普通的，包括DuoCL
    #train_model(model, train_loader, val_loader, epochs=200, lr=0.0001,device=device,outdir=outdir,tf='FT')
    #train_model_step_in_epoch(model, train_loader, val_loader, epochs=200, lr_freq=0.0001,lr_time=0.0001,device=device,outdir=outdir)
    #train_model_step(model, train_loader, val_loader, epochs_freq=50,epochs_time=200, lr=0.0001,device=device,outdir=outdir)
    train_TF_step(model, train_loader, val_loader, epochs_freq=200,epochs_time=200,epochs_fusion=200, lr_freq=0.0001,lr_time=0.0001,lr_fu=0.0001,device=device,
                  outdir=outdir)#,time_path='D:/ly123/Models/35_Conv_r_2/model_saved/model_epoch_200.pth',freq_path='D:/ly123/Models/36_CompCNN_r_2/model_saved/model_epoch_200.pth'
    #train_TF_vae(model, train_loader, val_loader, epochs_freq=100,epochs_time=200, lr_freq=0.0001,lr_time=0.0001,device=device,outdir=outdir)
    #train_model_comp(model, train_loader, val_loader, epochs=200, lr=0.0001,device=device,outdir=outdir,loss='comp')


    # 测试模型
    #test_model(model, test_loader,device=device,outdir=outdir,tf='FT')
    test_model_TF(model, test_loader,device=device,outdir=outdir)
    #test_model_vae(model, test_loader,device=device,outdir=outdir)
    #test_model_vanilla(model, test_loader,device=device,outdir=outdir)
    #test_model_comp(model, test_loader,device=device,outdir=outdir,loss='comp')
    print(f"Experiment {i} completed.")
# 5. 主程序
'''if __name__ == "__main__":
    # 这里可以使用多线程或多进程来并行运行多个实验
    for i in range(-10, 1):
        main(i)
        print(f"Experiment {i} completed.")'''
if __name__ == "__main__":
    '''multiprocessing.set_start_method('spawn')  # 尤其在 Windows 下要设置
    pool = multiprocessing.Pool(processes=min(4, multiprocessing.cpu_count()))  # 设置进程数量，最多4个可同时跑

    indices = list(range(2, 3))  # i 从 -10 到 10
    pool.map(main, indices)

    pool.close()
    pool.join()'''
    main(2)