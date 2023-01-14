# rnn-lstm-multiple-regression
使用svr, mlp, rnn, lstm, am-lstm进行多元时间序列回归预测

## SVR,MLP,RNN,LSTM,AM-LSTM(带时间注意力机制LSTM)
每种方法执行xxx_prac.py就完事儿

23年在看这个小demo，说实话，当时考虑太少了
对时间序列预测任务要有敬畏之心。有时候看单步预测的曲线和真实值拟合地挺好，其实有可能啥都不是。一个最基本的baseline就是：拿前一天真实值作为下一天的预测值，会发现曲线拟合也很不错。其实就是引入了一天的滞后性，在很多场景下，这种滞后性会使得整个预测任务毫无价值！
