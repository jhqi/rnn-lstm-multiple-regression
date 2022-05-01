from torch.utils.data import Dataset
# import tqdm
import pandas as pd
import json


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


# # example
# conf_dict = {
#     'bin': 8,
#     'missing': -9999999.0,
#     'continuous_cols': []
# }
#
# traindf = pd.read_csv('')
# testdf = pd.read_csv('')
#
# c = Cutter(conf_dict['continuous_cols'],
#            bins=conf_dict['bin'], missing=conf_dict['missing'])
# c.fit(pd.concat([traindf, testdf], axis=0))  # 按行拼接起来
# c.transform(traindf)
# c.transform(testdf)
