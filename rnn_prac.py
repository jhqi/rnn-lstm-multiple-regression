import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.data import DataLoader
from torch import nn
from utils import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决符号无法显示


seed = 991217
setup_seed(seed)

# file_name=['61800400','61801700','67066000']
file_name = ['61800400']


for tmp_file_name in file_name:
    tmp_file_path = os.path.join('./datas', tmp_file_name+'.csv')
    df = pd.read_csv(tmp_file_path)

    cfg = {
        'TIME_STEP': 30,
        'BATCH_SIZE': 64,
        'BINS': 6,
        'num_layers': 1,  # rnn层数
        'hidden_dim': 16,  # rnn隐藏层大小
        'lr': 0.005,
    }

    continuous_cols = ['prcp', 'RH', 'tmax', 'tmin', 'vp']

    c = Cutter(continuous_cols, bins=cfg['BINS'], missing=32700)
    c.fit(df)
    c.transform(df)

    # 划分数据集
    train_size = int(len(df) * 0.75)
    traindf = df.iloc[:train_size, :]
    testdf = df.iloc[train_size:, :]

    # discharge的均值和标准差，只用训练数据计算
    dis_mean = np.mean(traindf['discharge'].values)
    dis_std = np.std(traindf['discharge'].values)

    X, Y = get_input(df, cfg['TIME_STEP'], cfg['TIME_STEP'],
                     train_size - 1, dis_mean, dis_std)
    train_dataset = MyDataset(X, Y)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg['BATCH_SIZE'], shuffle=True)

    X, Y = get_input(df, cfg['TIME_STEP'], train_size,
                     len(df) - 1, dis_mean, dis_std)
    test_dataset = MyDataset(X, Y)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=cfg['BATCH_SIZE'], shuffle=False)

    input_dim = 6
    output_dim = 1

    model = MyRNN(input_dim=input_dim,
                  hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers'], output_dim=output_dim)
    model = model.cuda()
    loss_func = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    EPOCHS = 300

    # 训练
    min_test_loss = float(5)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0
        train_num = 0
        for X, y in train_loader:
            train_num += len(y)
            X = X.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_func(y_pred.view(-1), y.view(-1))
            # 反向传播
            loss.backward()
            # 将参数更新至网络中
            optimizer.step()
            total_train_loss += loss.item() * len(y)

        test_mean_loss = test_step(model, test_loader, loss_func)
        train_mean_loss = total_train_loss / train_num

        if train_mean_loss < 0.2 and test_mean_loss < min_test_loss:
            min_test_loss = test_mean_loss
            torch.save(model, './rnn_model.pth')

        print(f"epoch:{epoch}, train_mean_loss:{train_mean_loss}, test_mean_loss={test_mean_loss}")


    x_all = list(df['date'].values)
    y_all = df['discharge'].values

    model = torch.load('./rnn_model.pth', map_location=torch.device('cuda'))
    model.eval()

    # 检验训练集的拟合
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=False)
    train_pred = get_pred(model, train_loader)
    tmp_df = df['date']
    x_train_pred = list(tmp_df.iloc[cfg['TIME_STEP']:train_size].values.reshape(-1))
    y_train_pred = train_pred.numpy().reshape(-1)
    y_train_pred = y_train_pred * dis_std + dis_mean

    test_pred = get_pred(model, test_loader)
    x_test_pred = list(tmp_df.iloc[train_size:].values)
    y_test_pred = test_pred.numpy().reshape(-1)
    y_test_pred = y_test_pred * dis_std + dis_mean

    plt.figure(dpi=300, figsize=(24, 10))
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(20))
    plt.plot(x_all, y_all, color='blue', linewidth=3, alpha=0.3, label='真实值')
    plt.plot(x_train_pred, y_train_pred, color='red', linewidth=1, label='训练集-拟合')
    plt.plot(x_test_pred, y_test_pred, color='green', linewidth=1, label='测试集-预测')
    plt.legend(fontsize=16)
    # plt.title('Predict result of discharge',fontsize=24)
    plt.xticks(rotation=45)
    plt.xlabel('日期',fontsize=20)
    plt.ylabel('水流量',fontsize=20)

    plt.savefig('./fig_res/rnn_'+tmp_file_name+'.png')
    plt.close('all')

    #计算指标
    error_dic=calc_error(y_test_pred.reshape(-1),y_all[train_size:].reshape(-1))
    error_df=pd.DataFrame(error_dic,index=[0])
    error_df.to_csv('./error_value/rnn_'+tmp_file_name+'.csv',index=False)

    #写入测试集的真实结果，重复就重复吧
    test_gt_dic={
        "date":x_all[train_size:],
        "discharge":list(y_all[train_size:])
    }
    test_gt_df=pd.DataFrame(test_gt_dic)
    test_gt_df.to_csv('./pred_value/'+tmp_file_name+'_gt.csv',index=False)

    #写入测试集的预测结果
    pred_value_dic={
        "date":x_all[train_size:],
        "discharge":list(y_test_pred)
    }
    pred_value_df=pd.DataFrame(pred_value_dic)
    pred_value_df.to_csv('./pred_value/'+tmp_file_name+'_rnn.csv',index=False)

    # plt.show()
