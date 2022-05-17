import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils import setup_seed, Cutter
import pandas as pd
import os
from utils import calc_error
plt.rcParams['font.sans-serif']=['SimHei'] #解决中文显示
plt.rcParams['axes.unicode_minus'] = False #解决符号无法显示


seed=991217
setup_seed(seed)

file_name=['61800400','61801700','67066000']

for tmp_file_name in file_name:
    tmp_file_path=os.path.join('./datas',tmp_file_name+'.csv')
    df = pd.read_csv(tmp_file_path)

    continuous_cols=['prcp','RH','tmax','tmin','vp']
    bins=6
    missing=32700

    c = Cutter(continuous_cols, bins=bins, missing=missing)
    c.fit(df)
    c.transform(df)

    train_size = int(len(df) * 0.75)
    traindf = df.iloc[:train_size, :]
    testdf = df.iloc[train_size:, :]

    dis_mean = np.mean(traindf['discharge'].values)
    dis_std = np.std(traindf['discharge'].values)

    train_X=traindf.iloc[:,1:-1].values
    train_Y=traindf.iloc[:,-1].values.reshape(-1)
    test_X=testdf.iloc[:,1:-1].values
    test_Y=testdf.iloc[:,-1].values.reshape(-1)

    train_Y=(train_Y-dis_mean)/dis_std
    test_Y=(test_Y-dis_mean)/dis_std

    svr_rbf = SVR(kernel='rbf', C=1)
    svr_poly = SVR(kernel='poly', C=1)

    svr_rbf.fit(train_X, train_Y)
    y_rbf_train=svr_rbf.predict(train_X)
    y_rbf_test=svr_rbf.predict(test_X)

    svr_poly.fit(train_X, train_Y)
    y_poly_train=svr_poly.predict(train_X)
    y_poly_test=svr_poly.predict(test_X)

    x_all = list(df['date'].values)
    y_all = df['discharge'].values

    tmp_df = df['date']
    x_train_pred = list(tmp_df.iloc[:train_size].values.reshape(-1))
    y_train_pred = y_rbf_train * dis_std + dis_mean


    x_test_pred = list(tmp_df.iloc[train_size:].values)
    y_test_pred = y_rbf_test * dis_std + dis_mean

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

    plt.savefig('./fig_res/svr_'+tmp_file_name+'.png')
    plt.close('all')


    #计算指标
    error_dic=calc_error(y_test_pred.reshape(-1),y_all[train_size:].reshape(-1))
    error_df=pd.DataFrame(error_dic,index=[0])
    error_df.to_csv('./error_value/svr_'+tmp_file_name+'.csv',index=False)

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
    pred_value_df.to_csv('./pred_value/'+tmp_file_name+'_svr.csv',index=False)

    # plt.show()