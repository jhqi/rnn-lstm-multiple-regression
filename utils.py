import numpy as np
import torch
import pandas as pd
import random
from torch.utils.data import Dataset
from torch import nn
import json
import math

class Cutter(Dataset):
    def __init__(self, cols, bins=8, missing=32700, cut_result_path='cut.json'):
        self.cols = cols
        self.bins = bins
        self.missing = missing
        self.cut_result_path = cut_result_path

    def fit(self, df):
        dic = {}
        for col in self.cols:
            cur_series = df[col]
            cur_series = cur_series[cur_series != self.missing]  # 仅取非缺失值
            qcut_ret = pd.qcut(cur_series, self.bins,
                               labels=False, retbins=True, duplicates='drop')
            dic[col] = qcut_ret[1].tolist()
        json_str = json.dumps(dic, indent=4)
        with open(self.cut_result_path, 'w') as json_file:
            json_file.write(json_str)

    def transform(self, df):
        with open(self.cut_result_path, 'r') as f:
            json_f = json.load(f)
        for col in self.cols:
            save_num_bins = json_f[col]
            q_res = pd.cut(df[col], save_num_bins, labels=range(0, len(save_num_bins) - 1),
                           retbins=True, include_lowest=True)
            df[col] = q_res[0].cat.add_categories(
                len(save_num_bins) - 1).fillna(len(save_num_bins) - 1).astype('int')



def get_input(df, time_step, st, ed, dis_mean, dis_std):
    # data是二维df，要转成numpy，并对label标准化, feature已做分桶，无需再处理，第一列是日期，没用。最后一列是label
    # st 起始时刻，含，制作的第一个时序样本，其标签是df中st行的label
    # ed 截至时刻，含，制作的最后一个样本，其标签是df中下标为ed行的label
    # 输出: X:num_samples*time_step*6, dtype=np.float32
    #     Y: (num_samples,)
    # num_samples=ed-st+1

    assert st >= time_step, 'st前的样本数量不足！'
    assert ed < len(df), 'ed越界！'
    num_samples = ed - st + 1
    data = df.iloc[:, 1:].values.astype(np.float32)
    data[:, -1] = (data[:, -1] - dis_mean) / dis_std

    X = np.zeros((num_samples, time_step, 6), dtype=np.float32)
    Y = np.zeros((num_samples, 1), dtype=np.float32).reshape(-1)

    for i in range(num_samples):
        X[i, :, :] = data[i + st - time_step:i + st, :]
        Y[i] = data[i + st, -1]

    return X, Y


class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x = self.X[index, :, :]
        y = self.Y[index].reshape(-1)
        return x, y

    def __len__(self):
        return self.X.shape[0]


class MyRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MyRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_dim).requires_grad_().cuda()
        out, hn = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out


class MyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MyLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_dim).requires_grad_().cuda()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = False


def test_step(model, test_loader, loss_func):
    model.eval()
    test_num = 0
    total_test_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            test_num += len(y)
            X = X.cuda()
            y = y.cuda()
            pred = model(X)
            loss = loss_func(pred.view(-1), y.view(-1))
            total_test_loss += loss.item() * len(y)
    return total_test_loss / test_num


def get_pred(model, loader):
    model.eval()
    pred = torch.tensor([], dtype=torch.float32)
    with torch.no_grad():
        for X, y in loader:
            X = X.cuda()
            y_pred = model(X)
            pred = torch.cat((pred, y_pred.cpu()))
    return pred


class MyLSTM_AM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, fc_dropout, predict_len,time_step):
        super(MyLSTM_AM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.predict_len = predict_len
        self.time_step=time_step
        self.mysoftmax = nn.Softmax(dim=1)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstmcell = nn.LSTMCell(
            input_size=hidden_dim, hidden_size=hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(fc_dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def Atten(self, H):
        h = H[:, -1, :].unsqueeze(1)  # batch_size*1*hidden_dim
        H = H[:, -1-self.time_step:-1, :]  # batch_size*time_step*hidden_dim
        atten = self.mysoftmax(torch.matmul(
            h, H.transpose(1, 2)).transpose(1, 2))
        atten_H = torch.sum(atten*H, dim=1).unsqueeze(1)
        return torch.cat((atten_H, h), 2).squeeze(1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_dim).requires_grad_().cuda()
        H, (h, c) = self.lstm(x, (h0.detach(), c0.detach()))
        h = h.squeeze(0)
        c = c.squeeze(0)
        H_pre = torch.empty(
            (h.shape[0], self.predict_len, self.hidden_dim*2)).cuda()
        for i in range(self.predict_len):
            h_t, c_t = self.lstmcell(h, (h, c))
            H = torch.cat((H, h_t.unsqueeze(1)), 1)
            h_atten = self.Atten(H)
            H_pre[:, i, :] = h_atten
            h, c = h_t, c_t

        return self.fc(H_pre).squeeze(2)


def calc_error(pred,gt):
    #输入numpy
    mse=float(np.mean((pred-gt)**2))
    rmse=math.sqrt(mse)
    mae=float(np.mean(np.abs(pred-gt)))
    mape=float(np.mean(np.abs(pred-gt)/gt))
    return {
        "rmse":rmse,
        "mae":mae,
        "mape":mape
    }
