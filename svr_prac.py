import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils import setup_seed, Cutter
import pandas as pd

seed=7
setup_seed(seed)

df=pd.read_csv('./datas/61800400.csv')

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
y_rbf_train = y_rbf_train * dis_std + dis_mean
y_poly_train = y_poly_train * dis_std + dis_mean


x_test_pred = list(tmp_df.iloc[train_size:].values)
y_rbf_test = y_rbf_test * dis_std + dis_mean
y_poly_test = y_poly_test * dis_std + dis_mean

plt.figure(dpi=300, figsize=(24, 10))
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(20))
plt.plot(x_all, y_all, color='blue', linewidth=3, alpha=0.3, label='true')
plt.plot(x_train_pred, y_rbf_train, color='red', linewidth=1, label='train_fit_rbf')
# plt.plot(x_train_pred, y_poly_train, color='green', linewidth=1, label='train_fit_poly')
plt.plot(x_test_pred, y_rbf_test, color='green', linewidth=1, label='test_pred_rbf')
# plt.plot(x_test_pred, y_poly_test, color='gray', linewidth=1, label='test_pred_poly')


plt.legend()
plt.title('Predict result of discharge',fontsize=24)
plt.xticks(rotation=45)
plt.xlabel('date',fontsize=20)
plt.ylabel('discharge',fontsize=20)
plt.savefig('./svr_res.png')
# plt.show()