# 多进程-多线程数据读取工具类
# 以V2版本为基础，修改为双精度的数据集
import datetime
import json
import math
import os
import platform
import random
import time
import numpy as np
import torch
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import Dataset, DataLoader


def timestr(d=0):
    if d == 0:
        return time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
    elif d == 1:
        return time.strftime("%Y%m%d", time.localtime())
    else:
        return time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())


def is_valid_trade_date(date_str, fmtstr='%Y%m%d'):
    """
    判断日期字符串是否是有效的交易日期
    :param date_str:
    :param fmtstr:
    :return:
    """
    date_obj = datetime.datetime.strptime(str(date_str), fmtstr)
    weekday = date_obj.weekday()
    if weekday == 5 or weekday == 6:
        return False
    return True


def get_sub_data_list(all_dates, s, e):
    """
    获取子数据列表
    :param all_dates: 所有日期列表
    :param s: 开始日期
    :param e: 结束日期
    :return: 子数据列表
    """
    try:
        si = all_dates.index(s)
        ei = all_dates.index(e)
    except ValueError:
        print(f"日期{s}或{e}不存在于数据集中")
        return []
    if si > ei:
        print(f"开始日期{s}大于结束日期{e}")
        return []
    return all_dates[si:ei + 1]


def make_trade_date_list(start_date, end_date, datefmt='%Y%m%d'):
    # 将字符串转换为datetime对象
    start = datetime.datetime.strptime(str(start_date), datefmt)
    end = datetime.datetime.strptime(str(end_date), datefmt)

    # 存储有效日期的列表
    valid_dates = []

    # 当前日期从起始日期开始
    current_date = start

    # 循环直到当前日期超过结束日期
    while current_date <= end:
        # 检查当前日期是否是周六或周日
        if current_date.weekday() < 5:  # 0-4 是周一到周五
            # 将日期添加到列表中
            valid_dates.append(int(current_date.strftime(datefmt)))
        # 移动到下一天
        current_date += datetime.timedelta(days=1)

    return valid_dates


def read_record_from_binary(binary_file_path, field_count, read_index, record_count=1):
    """
    从二进制文件中读取第read_index条记录开始的连续record_count条记录
    :param binary_file_path:  二进制文件路径
    :param field_count:       每条记录的字段数
    :param read_index:        要读取的第1条记录的索引下标（从0开始）
    :param record_count:      要读取的记录数
    :return: [[]]
    """

    with open(binary_file_path, 'rb') as binary_file:
        # 跳转到第m条记录的起始位置
        binary_file.seek(field_count * read_index * 8, 0)  # 每个浮点数占用8个字节
        data = binary_file.read(record_count * field_count * 8)
        record_set = np.frombuffer(data, dtype=np.float64).reshape(record_count, field_count)
        return record_set


def read_one_stock_seq(data_path_bin, stock_code, start_idx, data_file_fields, need_seq_len):
    """
    读取一只股票序列
    :param data_path_bin: 数据文件路径
    :param stock_code: 股票代码int
    :param start_idx: 开始索引
    :param data_file_fields: 数据文件字段数
    :param need_seq_len: 序列长度L
    :return: 股票序列[L,C]
    """

    # 股票代码转换为字符串，并在前面补零，使其长度为6
    stock_code_str = str(stock_code).zfill(6)
    file = os.path.join(data_path_bin, f"{stock_code_str}.bin")
    data_bin = read_record_from_binary(file, data_file_fields, start_idx, need_seq_len)
    data_bin = torch.tensor(data_bin, dtype=torch.float64)
    return data_bin


def cut_stock_list(stock_list, group_num):
    """
    将股票列表切分成多个组
    :param stock_list: 股票列表
    :param group_num: 最多分成的组数
    :return: 组字典
    """

    group_dict = {}
    group_size = len(stock_list) // group_num
    if group_size <= 1:
        group_size = 1
    for i in range(group_num):
        if i == group_num - 1:
            # 最后一组
            stock_group = stock_list[i * group_size:]
        else:
            stock_group = stock_list[i * group_size: (i + 1) * group_size]
        group_dict[i] = stock_group
    return group_dict


class DatasetConfig:
    def __init__(self):
        # 数据集配置字典，存储多个数据集的配置信息
        self.config = {
            "xt_day_39f": {
                "data_path_bin": '/data/yy_data/xt_day/d_data/bin_data',
                "index_file": '/data/yy_data/xt_day/d_data/index.json',
                "scaler_file": '/data/yy_data/xt_day/d_data/scaler_info.txt',
                "channels": 39,
                "period": 'day',
            },
            "ts_day_54f": {
                "data_path_bin": '/data/yy_data/tushare_data/54f_data/bin_data',
                "index_file": '/data/yy_data/tushare_data/54f_data/index.json',
                "scaler_file": '/data/yy_data/tushare_data/54f_data/scaler_info.txt',
                "channels": 54,
                "period": 'day',
            },
            "xt_5m_52f": {
                "data_path_bin": '/data/yy_data/five_minute_data/52f_data/bin_data',
                "index_file": '/data/yy_data/five_minute_data/52f_data/index.json',
                "scaler_file": '/data/yy_data/five_minute_data/52f_data/scaler_info.txt',
                "channels": 52,
                "period": 'five_minute',
            },
            "xt_5m_52f_1129": {
                "data_path_bin": '/data/yy_data/five_minute_data/52f_data_1129/bin_data',
                "index_file": '/data/yy_data/five_minute_data/52f_data_1129/index.json',
                "scaler_file": '/data/yy_data/five_minute_data/52f_data_1129/scaler_info.txt',
                "channels": 52,
                "period": 'five_minute',
            },
            "xt_day_minute_39f": {
                "data_path_bin": '/data/yy_data/xt_day_minute/bin_data',
                "index_file": '/data/yy_data/xt_day_minute/index.json',
                "scaler_file": '/data/yy_data/xt_day/d_data/scaler_info.txt',
                "channels": 39,
                "period": 'day',
            },
            "xt_day_all_21f": {
                "data_path_bin": '/data/yy_data/xt_hq_data/d_data/bin_data',
                "index_file": '/data/yy_data/xt_hq_data/d_data/index.json',
                "scaler_file": '/data/yy_data/xt_hq_data/d_data/scaler_info.txt',
                "channels": 21,
                "period": 'day',
            },
            "ts_day_54f_ebj": {
                "data_path_bin": '/mnt/yy_data/tushare_data/54f_ebj_data/bin_data',
                "index_file": '/mnt/yy_data/tushare_data/54f_ebj_data/index.json',
                "scaler_file": '/mnt/yy_data/tushare_data/54f_ebj_data/scaler_info.txt',
                "channels": 54,
                "period": 'day',
            },
            "ts_day_calendar_54f": {
                "data_path_bin": '/data/yy_data/tushare_data/54f_calendar_data/bin_data',
                "index_file": '/data/yy_data/tushare_data/54f_calendar_data/index.json',
                "scaler_file": '/data/yy_data/tushare_data/54f_calendar_data/scaler_info.txt',
                "channels": 54,
                "period": 'day',
            },
            "59f_std_data_d": {
                "data_path_bin": '/data/yy_data/tushare_data/59f_std_data_d/bin_data',
                "index_file": '/data/yy_data/tushare_data/59f_std_data_d/index.json',
                "scaler_file": '/data/yy_data/tushare_data/59f_std_data_d/scaler_info.txt',
                "channels": 59,
                "period": 'day',
        },
            "41f_5m_std_data_d": {
                "data_path_bin": '/data/yy_data/five_minute_data/41f_5m_std_data_d/bin_data',
                "index_file": '/data/yy_data/five_minute_data/41f_5m_std_data_d/index.json',
                "scaler_file": '/data/yy_data/five_minute_data/41f_5m_std_data_d/scaler_info.txt',
                "channels": 41,
                "period": 'five_minute',
        },
            "ts_day_54f_d": {
                "data_path_bin": '/data/yy_data/tushare_data/54f_data_d/bin_data',
                "index_file": '/data/yy_data/tushare_data/54f_data_d/index.json',
                "scaler_file": '/data/yy_data/tushare_data/54f_data_d/scaler_info.txt',
                "channels": 54,
                "period": 'day',
        },
            "ts_day_55f_avg_price": {
                "data_path_bin": '/data/yy_data/tushare_data/55f_avg_price/bin_data',
                "index_file": '/data/yy_data/tushare_data/55f_avg_price/index.json',
                "scaler_file": '/data/yy_data/tushare_data/55f_avg_price/scaler_info.txt',
                "channels": 55,
                "period": 'day',
            },
            "ts_day_index_60f": {
                "data_path_bin": '/data/raw_generated_data/tushare_data/60f_index_data/bin_data',
                "index_file": '/data/raw_generated_data/tushare_data/60f_index_data/index.json',
                "scaler_file": '/data/raw_generated_data/tushare_data/60f_index_data/scaler_info.txt',
                "channels": 60,
                "period": 'day',
            },
        }
        if platform.system() == 'Windows':
            self.config["ts_day_54f_d"]["data_path_bin"] = 'D:/stock_data/54f_data_d/bin_data'
            self.config["ts_day_54f_d"]["index_file"] = 'D:/stock_data/54f_data_d/index.json'
            self.config["ts_day_54f_d"]["scaler_file"] = 'D:/stock_data/54f_data_d/scaler_info.txt'

            self.config["ts_day_55f_avg_price"]["data_path_bin"] = 'D:/stock_data/55f_avg_price/bin_data'
            self.config["ts_day_55f_avg_price"]["index_file"] = 'D:/stock_data/55f_avg_price/index.json'
            self.config["ts_day_55f_avg_price"]["scaler_file"] = 'D:/stock_data/55f_avg_price/scaler_info.txt'

            self.config["ts_day_index_60f"]["data_path_bin"] = 'D:/stock_data/60f_index_data/bin_data'
            self.config["ts_day_index_60f"]["index_file"] = 'D:/stock_data/60f_index_data/index.json'
            self.config["ts_day_index_60f"]["scaler_file"] = 'D:/stock_data/60f_index_data/scaler_info.txt'


    def get_config(self, dataset_name):
        """
        根据数据集名称获取配置，并自动验证文件和文件夹的有效性
        """
        if dataset_name not in self.config:
            raise ValueError(f"数据集 {dataset_name} 配置不存在")

        config = self.config[dataset_name]

        # 在获取配置时，自动校验文件和文件夹
        self._validate_file(config["index_file"])
        self._validate_file(config["scaler_file"])
        self._validate_directory(config["data_path_bin"])

        return config

    def _validate_file(self, file_path):
        """
        校验文件是否存在
        """
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在！")
            raise FileNotFoundError(f"文件 {file_path} 不存在")

    def _validate_directory(self, dir_path):
        """
        校验文件夹是否存在且不为空
        """
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"文件夹 {dir_path} 不存在")
        if not os.listdir(dir_path):
            raise ValueError(f"文件夹 {dir_path} 为空")


class DiskBinDataset(Dataset):
    # 从磁盘文件读取数据的Dataset
    def __init__(self, args, ds, dict_stock, dict_date, data_set_dict, all_dates_index, period, transform=None):
        """
        初始化数据集
        :param args: 参数
        """
        self.args = args
        self.ds = ds  # 数据集
        self.dict_stock = dict_stock  # 股票索引字典 {股票代码: {日期: 所在行序号}}
        self.dict_date = dict_date  # 日期索引字典 {日期: {股票代码，所在行序号}}
        self.data_set_dict = data_set_dict  # {日期, [股票代码列表]}
        self.all_dates_index = all_dates_index  # 日期索引字典 {日期: 所在行序号}
        self.transform = transform  # 转换操作
        self.period = period  # 周期

    def __len__(self):
        """
        返回数据集中样本的数量
        """
        return len(self.ds)

    def __getitem__(self, idx):
        """
        根据索引idx获取一个样本
        """
        # 根据idx从磁盘读取数据
        day = self.ds[idx][0]
        group_idx = self.ds[idx][1]
        if len(self.ds[idx]) == 3:
            minute_idx = self.ds[idx][2]
            data = self.get_sism_sample(day, group_idx, minute_idx)
        else:
            data = self.get_sism_sample(day, group_idx)

        # 应用转换操作
        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_sism_sample(self, day, group_idx, minute_idx=None):
        """
        获取一个样本的数据
        :param day 日期
        :param group_idx 组索引
        :return: input_seq[M,L,C], future_seq[M,P,C], global_seq[G,L,C]
        """

        si_stock_list, sg_stock_list = self.get_si_stock_list(day, group_idx)
        # print("SI股票列表：", si_stock_list)

        i_seq = self.read_si_seq(si_stock_list, day, minute_idx, self.args.time_step + self.args.future_step, )  # [M,L+P,C]
        if self.args.future_step > 0:
            f_seq = i_seq[:, -self.args.future_step:, :]  # [M,P,C]
            i_seq = i_seq[:, :-self.args.future_step, :]  # [M,L,C]
        else:
            f_seq = []
        if sg_stock_list:
            g_seq = self.read_si_seq(sg_stock_list, day, minute_idx, self.args.time_step)  # [G,L,C]
        else:
            g_seq = []

        return i_seq, f_seq, g_seq

    def read_si_seq(self, stock_list, start_date_key, minute_idx, need_seq_len):
        """
        读取一组股票序列
        :param stock_list: 股票列表S
        :param start_date_key: 开始日期
        :param need_seq_len: 序列长度L
        :return: 股票序列[S,L,C]_
        """

        # 单线程读取
        seq = []
        for stock_code in stock_list:
            stock_first_date = self.dict_stock[stock_code][0]
            si = self.all_dates_index[stock_first_date]  # 股票上市日期在全体日期列表中的索引
            ei = self.all_dates_index[start_date_key]  # 现在要读取的时序的首个日期在全体日期列表中的索引
            start_idx = ei - si  # 计算得到要读取的数据在该股票文件中的偏移位置

            # 如果是5分钟数据，需要乘48
            if self.period == 'five_minute':
                start_idx = start_idx * 48
                start_idx += minute_idx

            # 读取一只股票的数据
            stock_data = read_one_stock_seq(self.args.data_path_bin, stock_code, start_idx,
                                            self.args.channels,
                                            need_seq_len)  # [L,C]
            seq.append(stock_data)
        seq = torch.stack(seq, dim=0)  # [S,L,C]
        return seq

    def get_si_stock_list(self, day, group_idx):
        """
        获取一个组的股票列表
        :param day: 起始日期
        :param group_idx: 组号
        :return: 股票列表S
        """

        # 根据日期找到对应的股票列表
        stock_list = self.data_set_dict[day]
        total_count = len(stock_list)
        start_idx = group_idx * self.args.sicount
        stock_list_cp = stock_list.copy()
        # random.shuffle(stock_list_cp)
        if start_idx + self.args.sicount > total_count:
            # 返回最后的self.args.sicount个股票
            return stock_list[-self.args.sicount:], stock_list_cp[:self.args.sgcount]
        else:
            return stock_list[start_idx:start_idx + self.args.sicount], stock_list_cp[:self.args.sgcount]


class DiskStockDataReader:
    def __init__(self, args):
        self.args = args

        # 配置数据集的配置
        dc = DatasetConfig()
        data_config = dc.get_config(args.data_name)
        self.args.data_path_bin = data_config.get('data_path_bin')
        self.args.index_file = data_config.get('index_file')
        self.args.scaler_file = data_config.get('scaler_file')
        self.args.channels = data_config.get('channels')
        self.args.period = data_config.get('period')

        self.dict_stock = {}  # 股票索引字典 {股票代码: {日期: 所在行序号}}
        self.dict_date = {}  # 日期索引字典 {日期: {股票代码，所在行序号}}
        self.data_set_dict = None  # {日期, [股票代码列表]}
        self.all_dates_list = []
        self.all_dates_index = {}
        need_seq_len = self.args.time_step + self.args.future_step  # 输入序列长度+预测序列长度
        if self.args.period == 'five_minute':
            self.need_day_len = math.ceil(need_seq_len / 48) + 1
        elif self.args.period == 'day':
            self.need_day_len = need_seq_len
        else:
            raise ValueError("period目前只支持five_minute和day")

        self.init()  # 初始化数据

    def init(self):
        # 加载索引文件
        print(f"{timestr()}初始化数据加载器。。。")
        time_start = time.time()
        self.dict_stock, self.dict_date = self.load_index_from_file()
        si = len(self.dict_stock)
        print(f"股票数量：{si}")
        di = len(self.dict_date)
        date_count = sum([len(v) for v in self.dict_date.values()])
        print(f"日期数量：{di}, 总数据量：{date_count}")
        keys = list(self.dict_date.keys())
        print(f"前10个日期列表：{keys[:10]}")
        print(f"后10个日期列表：{keys[-10:]}")

        # 创建数据索引一级字典
        print(f"{timestr()}创建数据索引一级字典。。。")
        self.data_set_dict = self.prepare_data_dict()
        print(f"{timestr()}重新洗牌数据索引一级字典。。。")
        self.re_shuffle_data_dict()
        print(f"{timestr()}数据索引一级字典创建完成")
        print(f"{timestr()}数据加载器初始化完成，耗时{time.time() - time_start:.2f}秒")

    def get_next_date(self, start_date, trade_days):
        """
        已知一个起始日期,计算在经过trade_days个交易日后的下一个交易日
        :param start_date: 起始日期
        :param trade_days: 交易日数
        :return: 未来的交易日
        """

        si = self.all_dates_index[start_date]
        ei = si + trade_days
        if ei >= len(self.all_dates_list):
            return -1
        return self.all_dates_list[ei]

    def load_index_from_file(self):
        """
        从索引文件中加载索引
        :return: dict_stock, dict_date
        """

        # 加载索引文件
        print(f"{timestr()}从文件{self.args.index_file}加载股票索引文件。。。")
        with open(self.args.index_file, 'r', encoding='utf-8') as file:
            dict_stock = json.load(file)

        # 由于从json文件中加载字典时，所有的key都是字符串，因此需要将key转换为int类型
        # 在V2版本中，这里的value是一个list，只有两个元素，分别是首末日期int
        dict_stock = {int(k): v for k, v in dict_stock.items()}

        # 根据股票索引来创建日期索引，方便后续根据日期来查找股票
        dict_date = {}  # 日期索引字典 {日期: [股票代码，所在行序号]}
        # 汇聚所有的股票代码索引，得到日期索引
        print(f"{timestr()}创建日期索引。。。")
        count = 0

        # 遍历所有股票代码索引，得到全体数据集的最小日期和最大日期
        min_date = 99999999
        max_date = 0
        for stock_code, stock_index in dict_stock.items():
            # 遍历所有股票代码索引，得到股票代码stock_code和首末日期date
            # 生成首末日期之间的所有日期列表
            if stock_index[0] < min_date:
                min_date = stock_index[0]
            if stock_index[1] > max_date:
                max_date = stock_index[1]
        # 生成全部日期列表(!!!注意，这里的日期是有序的，而且不包含周末！！！)
        self.all_dates_list = make_trade_date_list(min_date, max_date, '%Y%m%d')
        # 初始化日期索引字典
        # 创建日期索引字典用于加快查询速度
        self.all_dates_index = {}
        idx = 0
        for date in self.all_dates_list:
            dict_date[date] = []
            self.all_dates_index[date] = idx
            idx += 1

        for stock_code, stock_index in dict_stock.items():
            # 遍历所有股票代码索引，得到股票代码stock_code和首末日期date
            # 生成首末日期之间的所有日期列表
            si = self.all_dates_index[stock_index[0]]
            ei = self.all_dates_index[stock_index[1]]
            online_dates = self.all_dates_list[si:ei + 1]
            for day in online_dates:
                dict_date[day].append(stock_code)  # 设定该日期包含这支股票
            count += 1

        # dict_date的key是日期已经是排序过的，不需要再排序
        print(f"{timestr()}创建日期索引，DONE")

        return dict_stock, dict_date

    def prepare_data_dict(self):
        """
        创建数据索引一级字典
        :return: 股票索引字典 {日期: [股票代码列表]}   这里每个日期所对应的股票代码是指符合要求能用于构造样本序列长度的所有股票
        """

        # 这里要准备的数据是一个日期-股票列表的字典
        # key为日期，value为股票列表,这个股票列表表示的是在这个日期上，可以用来构造样本的所有股票代码
        if self.data_set_dict is not None:
            return self.data_set_dict

        self.data_set_dict = {}

        # 遍历整个日期索引，看看哪一天作为起始日期时，是可以构造出完整的输入序列的
        # 生成的每个一样本索引是一个三元组[日期，[SI股票代码列表]，[SG股票代码列表]]
        date_key_list = list(self.dict_date.keys())  # 日期列表(这个日期已经是排序过的)

        date_key_len = len(date_key_list)

        if date_key_len < self.need_day_len:
            print(f"数据量不足，无法生成序列")
            return None
        for di in range(date_key_len - self.need_day_len + 1):
            # 遍历所有的合法日期
            start_day = date_key_list[di]
            end_day = date_key_list[di + self.need_day_len - 1]
            # 取出第一天在线的股票列表
            start_online_stocks = set(self.dict_date[start_day])
            if len(start_online_stocks) < self.args.sicount or len(start_online_stocks) < self.args.sgcount:
                continue
            # 取出最后一天在线的股票列表
            end_online_stocks = set(self.dict_date[end_day])
            if len(end_online_stocks) < self.args.sicount or len(end_online_stocks) < self.args.sgcount:
                continue
            # 取出首末两天同时都在线的股票列表
            online_stocks = start_online_stocks.intersection(end_online_stocks)
            if len(online_stocks) < self.args.sicount or len(online_stocks) < self.args.sgcount:
                continue

            self.data_set_dict[start_day] = list(online_stocks)

        return self.data_set_dict

    @classmethod
    def shuffle_day_data(self, day_data):
        random.shuffle(day_data)
        return day_data

    def re_shuffle_data_dict(self):
        """
        重新洗牌数据索引一级字典
        :return: None
        """

        # 重新洗牌数据索引一级字典, 因为data_loader本身会被shuffle,所以只需要把data_loader中所对应的那个组别ID
        # 所对应的一组股票代码重新排列即可
        # 由于实际上并没有保存每一组的股票代码，只是保存了组号，在读取时根据组号去取了对应那一组股票
        # 所以，re_shuffle操作，只需要把data_set_dict中每一天的股票列表重新洗牌即可
        # 而且这个data_set_dict是按引用传递给了所有的dataset实例，因此此处调用一次，则所有dataset实例都会被更新

        if self.data_set_dict is None:
            raise ValueError("数据索引一级字典为空，无法重新洗牌")
        # 遍历所有日期，重新洗牌股票列表
        # for day in self.data_set_dict.keys():
        #     random.shuffle(self.data_set_dict[day])

        with ProcessPoolExecutor(max_workers=5) as executor:
            self.data_set_dict = dict(zip(self.data_set_dict.keys(), executor.map(self.shuffle_day_data, self.data_set_dict.values())))

    def create_data_loader(self,start_time, end_time, stride=1, refresh=False):
        """
        创建数据加载器
        :return: data_loader
        """
        # 当前数据读取模式为SISM(需要返回input_seq,future_seq,global_seq)
        if refresh:
            self.re_shuffle_data_dict()

        time_start = time.time()
        print(f"{timestr()}创建data_loader。。。")

        # 当前时刻，数据一级索引已经生成了
        # self.data_set_dict {日期: [股票代码列表]} 这里记录了每天可用以构造样本的股票列表
        # 现在需要生成data_loader的二级索引，只需要一个日期，一个组别序号即可
        # 对于每一个日期，只需要计算出可以分成多少组就行了

        data_set = []
        # 遍历所有日期，构造样本
        data_set_keys = list(self.data_set_dict.keys())

        if self.args.period == 'day':
            for day_index in range(0, len(data_set_keys), stride):

                day = data_set_keys[day_index]
                if day < start_time:
                    continue

                di = self.all_dates_list.index(day)  # 算出该日期在全体日期列表中的索引
                actual_end_day = self.all_dates_list[di + self.need_day_len - 1]  # 获取时间窗口实际结束的天
                if actual_end_day > end_time:
                    continue

                stock_list = self.data_set_dict[day]
                day_stock_count = len(stock_list)  # 该日期作为起始日期时，可以构造出完整的输入序列的股票列表
                group_count = (day_stock_count + self.args.sicount - 1) // self.args.sicount  # 计算组数
                for i in range(group_count):
                    data_set.append([day, i])

        elif self.args.period == 'five_minute':
            for minute_index in range(0, len(data_set_keys) * 48, self.args.stride):
                day_index = minute_index // 48  # 算出天的索引
                day = data_set_keys[day_index]
                if day < start_time:
                    continue
                # 计算实际结束的天
                actual_end_day = self.get_next_date(day, self.need_day_len)
                if actual_end_day == -1:
                    break
                if actual_end_day > end_time:
                    continue
                day_minute_index = minute_index % 48  # 算出
                stock_list = self.data_set_dict[day]
                day_stock_count = len(stock_list)  # 该日期作为起始日期时，可以构造出完整的输入序列的股票列表
                group_count = (day_stock_count + self.args.sicount - 1) // self.args.sicount  # 计算组数
                for i in range(group_count):
                    data_set.append([day, i, day_minute_index])

        if not data_set:
            raise Exception("数据集为空")

        data_loader = DataLoader(
            DiskBinDataset(self.args, data_set, self.dict_stock, self.dict_date, self.data_set_dict,
                           self.all_dates_index, self.args.period),
            batch_size=self.args.batch_size, shuffle=self.args.shuffle,
            num_workers=self.args.num_workers, prefetch_factor=4)

        print(f"{timestr()}创建data_loader完成，耗时{time.time() - time_start:.2f}秒")
        return data_loader


class CheckData:
    def __init__(self, **kwargs):
        self.args = kwargs.get("args")
        self.scaler_info = {}
        # 配置数据集的配置
        dc = DatasetConfig()
        data_config = dc.get_config(self.args.data_name)
        self.args.data_path_bin = data_config.get('data_path_bin')
        self.args.index_file = data_config.get('index_file')
        self.args.scaler_file = data_config.get('scaler_file')
        self.args.channels = data_config.get('channels')
        self.args.period = data_config.get('period')

        with open(self.args.scaler_file, 'r', encoding='utf-8') as f:
            scaler_info = json.load(f)
        self.scaler_info = scaler_info
        self.field_list = list(self.scaler_info.keys())
        self.error_list = []

    # 还原原始数据
    def restore_original_data(self, batch_data, n=3):
        """
        验证数据：将归一化后的数据还原为原始数据
        :param n: 用于打印第一个数据样本的前n个时间步
        :param batch_data:
        :return:
        """
        torch.set_printoptions(sci_mode=False)

        batch_data_clone = batch_data.clone()
        if not self.scaler_info:
            print("还原原始数据需要传入scaler_file文件")
            return
        for filed, v in self.scaler_info.items():

            field_info = self.scaler_info[filed]
            if len(field_info) > 1:
                field_type, fmin, fmax, fmean, fstd = field_info[0], field_info[1], field_info[2], field_info[3], \
                field_info[4]
            else:
                field_type = field_info[0]

            i = self.field_list.index(filed)
            if field_type == 'MM-YY':
                # 年份字段只进行归一化(这里相当于min-max归一化, 1990-2030之间的值都归一化到0-1之间)
                batch_data_clone[:, :, :, i] = batch_data_clone[:, :, :, i] * 40 + 1990
            elif field_type == 'MM-MM':
                # 月份字段只进行归一化（这里相当于min-max归一化, 1-12之间的值都归一化到0-1之间）
                batch_data_clone[:, :, :, i] = batch_data_clone[:, :, :, i] * 12 + 1
            elif field_type == 'MM-DD':
                # 日期字段只进行归一化(这里相当于min-max归一化, 1-31之间的值都归一化到0-1之间)
                batch_data_clone[:, :, :, i] = batch_data_clone[:, :, :, i] * 31 + 1
            elif field_type == 'MM-WW':
                # 星期字段只进行归一化(这里相当于min-max归一化, 0~6之间的值都归一化到0-1之间)
                batch_data_clone[:, :, :, i] = batch_data_clone[:, :, :, i] * 6
            elif field_type == 'MM-MINUTE-5':
                # 分钟字段只进行归一化(这里相当于min-max归一化, 1-48之间的值都归一化到0-1之间)
                batch_data_clone[:, :, :, i] = batch_data_clone[:, :, :, i] * 48 + 1
            elif field_type == 'MM-EXCHANGE':
                batch_data_clone[:, :, :, i] = batch_data_clone[:, :, :, i] * 10
            elif field_type == 'MM-MARKET':
                batch_data_clone[:, :, :, i] = batch_data_clone[:, :, :, i] * 20
            elif field_type == 'MM-INDUSTRY':
                batch_data_clone[:, :, :, i] = batch_data_clone[:, :, :, i] * 200
            elif field_type == 'MMS-PRICE':
                # 价格字段需要进行标准化
                batch_data_clone[:, :, :, i] = batch_data_clone[:, :, :, i] * (fstd + 1e-8) + fmean
                batch_data_clone[:, :, :, i] = batch_data_clone[:, :, :, i] * (fmax - fmin + 1e-8) + fmin
            elif field_type == 'MMS':
                # 非价格字段需要进行标准化
                batch_data_clone[:, :, :, i] = batch_data_clone[:, :, :, i] * (fstd + 1e-8) + fmean
                batch_data_clone[:, :, :, i] = batch_data_clone[:, :, :, i] * (fmax - fmin + 1e-8) + fmin
            elif field_type == 'MMS-STD':
                # 非价格字段需要进行标准化
                batch_data_clone[:, :, :, i] = batch_data_clone[:, :, :, i] * (fstd + 1e-8) + fmean
            elif field_type == 'NORM-PRICE':
                batch_data_clone[:, :, :, i] = batch_data_clone[:, :, :, i] * 5000
            elif field_type == 'MM-LIMIT':
                # 这里相当于min-max归一化, 1~5之间的值都归一化到0-1之间
                batch_data_clone[:, :, :, i] = batch_data_clone[:, :, :, i] * 5 + 1
            else:
                continue

        print(batch_data_clone[0, 0, :n, :])
        return batch_data_clone

    # 验证数据细节
    def validate_detail(self, tensor_seq: torch.Tensor, start_time: int, end_time: int):
        shape = tensor_seq.shape
        for b in range(shape[0]):
            start_date_time_now = None
            end_date_time_now = None

            for s in range(shape[1]):
                assert tensor_seq[b, s, 0, 0] == tensor_seq[b, s, -1, 0], f"时间维度上股票编码不一致: b:{b} s:{s} {tensor_seq[b, s, 0, 0]} {tensor_seq[b, s, -1, 0]}"

                year_index = self.field_list.index('gen_year')
                month_index = self.field_list.index('gen_month')
                day_index = self.field_list.index('gen_day')
                if self.args.period == 'five_minute':
                    minute_index = self.field_list.index('gen_minute')
                year_scaler = self.scaler_info.get('gen_year')
                if year_scaler[0] == 'MMS-STD':
                    start_year = tensor_seq[b, s, 0, year_index] * (year_scaler[4] + 1e-8) + year_scaler[3]
                    end_year = tensor_seq[b, s, -1, year_index] * (year_scaler[4] + 1e-8) + year_scaler[3]
                else:
                    start_year = tensor_seq[b, s, 0, year_index] * 40 + 1990
                    end_year = tensor_seq[b, s, -1, year_index] * 40 + 1990

                month_scaler = self.scaler_info.get('gen_month')
                if self.scaler_info.get('gen_month')[0] == 'MMS-STD':
                    start_month = tensor_seq[b, s, 0, month_index] * (month_scaler[4] + 1e-8) + month_scaler[3]
                    end_month = tensor_seq[b, s, -1, month_index] * (month_scaler[4] + 1e-8) + month_scaler[3]
                else:
                    start_month = tensor_seq[b, s, 0, month_index] * 12 + 1
                    end_month = tensor_seq[b, s, -1, month_index] * 12 + 1

                day_scaler = self.scaler_info.get('gen_day')
                if self.scaler_info.get('gen_day')[0] == 'MMS-STD':
                    start_day = tensor_seq[b, s, 0, day_index] * (day_scaler[4] + 1e-8) + day_scaler[3]
                    end_day = tensor_seq[b, s, -1, day_index] * (day_scaler[4] + 1e-8) + day_scaler[3]
                else:
                    start_day = tensor_seq[b, s, 0, day_index] * 31 + 1
                    end_day = tensor_seq[b, s, -1, day_index] * 31 + 1

                if self.args.period == 'five_minute':
                    if self.scaler_info.get('gen_day')[0] == 'MMS-STD':
                        start_minute = tensor_seq[b, s, 0, minute_index] * (day_scaler[4] + 1e-8) + day_scaler[3]
                        end_minute = tensor_seq[b, s, -1, minute_index] * (day_scaler[4] + 1e-8) + day_scaler[3]
                    else:
                        start_minute = tensor_seq[b, s, 0, 8] * 48 + 1
                        end_minute = tensor_seq[b, s, -1, 8] * 48 + 1
                    start_date_time = int(f"{int(start_year):04d}{int(start_month):02d}{int(start_day):02d}{int(start_minute):02d}")
                    end_date_time = int(f"{int(end_year):04d}{int(end_month):02d}{int(end_day):02d}{int(end_minute):02d}")
                    assert start_time * 100 <= start_date_time and end_time * 100 >= end_date_time, "存在信息泄露"
                else:
                    start_date_time = int(f"{int(start_year):04d}{int(start_month):02d}{int(start_day):02d}")
                    end_date_time = int(f"{int(end_year):04d}{int(end_month):02d}{int(end_day):02d}")
                    assert start_time <= start_date_time and end_time >= end_date_time, "存在信息泄露"

                if start_date_time_now and start_date_time != start_date_time_now:
                    raise Exception(f"S和G股票维度上的时间存在不相等的情况: b:{b} s:{s} {start_date_time} {start_date_time_now}")

                if end_date_time_now and end_date_time != end_date_time_now:
                    raise Exception(f"S和G股票维度上的时间存在不相等的情况: b:{b} s:{s}{end_date_time} {end_date_time_now}")

                start_date_time_now = start_date_time
                end_date_time_now = end_date_time

    # 验证数据
    def validation_data(self, *args, **kwargs):
        """
        验证数据
        :param kwargs: input_seq, future_seq, global_seq, label

        input_seq [B,S,T,C]
        future_seq [B,S,F,C]
        global_seq [B,G,T,C]
        label [B,S,C']

        1. 检验input_seq, future_seq, global_seq, label的形状是否正确
        2. 检验时间维度上股票编码是否一致
        3. 检验股票维度上的时间是否一致
        4. 检验时间是否存在起始时间和结束时间范围内，防止信息泄露

        """
        print("开始验证数据。。。")
        # 检查每个batch的形状是否正确
        assert len(args) == 4, "填写input_seq, future_seq, global_seq, label"
        input_seq, future_seq, global_seq, label = args[0], args[1], args[2], args[3]

        start_time = kwargs.get("start_time", 0)
        end_time = kwargs.get("end_time", 99999999)

        c_input_seq_shape = [self.args.batch_size, self.args.sicount, self.args.time_step, self.args.channels]
        c_future_seq_shape = [self.args.batch_size, self.args.sicount, self.args.future_step, self.args.channels]
        c_global_seq_shape = [self.args.batch_size, self.args.sgcount, self.args.time_step, self.args.channels]

        input_seq_shape = list(input_seq.shape)
        if c_input_seq_shape != input_seq_shape:
            print(f"input_seq形状有误,{c_input_seq_shape} {input_seq_shape}")

        if isinstance(future_seq, torch.Tensor):
            future_seq_shape = list(future_seq.shape)
            if c_future_seq_shape != future_seq_shape:
                print(f"future_seq形状有误,{c_future_seq_shape} {future_seq_shape}")

        if isinstance(global_seq, torch.Tensor):
            global_seq_shape = list(global_seq.shape)
            if c_global_seq_shape != global_seq_shape:
                print(f"global_seq形状有误,{c_global_seq_shape} {global_seq_shape}")

        if isinstance(label, torch.Tensor):
            class_num = self.args.num_class_value.shape[0] + 1
            c_label_shape = [self.args.batch_size, self.args.sicount, class_num]
            if c_label_shape != list(label.shape):
                print("label状有误")

        # 验证数据的细节
        for i in [input_seq, future_seq, global_seq]:
            if isinstance(i, torch.Tensor):
                self.validate_detail(i, start_time, end_time)

        print("验证数据完成")

    def check_pad_data(self, pad_path, minute_num=48):
        """
        检查分钟的padding数据
        :param pad_path:
        :param minute_num:
        :return:
        """
        # 读取全部数据进入内存
        df = pd.read_csv(pad_path, header=None)
        year_idx = self.field_list.index('gen_year')
        month_idx = self.field_list.index('gen_month')
        day_idx = self.field_list.index('gen_day')
        minute_idx = self.field_list.index('gen_minute')
        # 用来存储缺失数据的列表
        error_list = []

        # 遍历所有数据行
        this_day = None
        minutes_seen = set()  # 用来记录当前日期中已出现的分钟数

        for row in df.itertuples():
            day_record = list(row[1:])
            # 格式化当前行的日期为 YYYYMMDD 格式
            now_day = f"{int(day_record[year_idx]):04d}{int(day_record[month_idx]):02d}{int(day_record[day_idx]):02d}"
            current_minute = int(day_record[minute_idx])

            # 如果我们遇到的是新的日期，则检查前一个日期的数据是否完整，并重置状态
            if now_day != this_day:
                if this_day is not None:
                    # 检查前一天的数据，是否缺少分钟数据
                    missing_minutes = set(range(1, minute_num)) - minutes_seen  # 计算缺失的分钟
                    for minute in missing_minutes:
                        error_list.append((this_day, minute))

                # 重置状态
                this_day = now_day
                minutes_seen = set()  # 清空已见分钟数

            # 将当前分钟记录在 minutes_seen 集合中
            minutes_seen.add(current_minute)

        # 最后检查最后一天的数据
        if this_day is not None:
            missing_minutes = set(range(1, minute_num)) - minutes_seen  # 计算缺失的分钟
            for minute in missing_minutes:
                error_list.append((this_day, minute))

        print(f"padding文件缺失的数据:{len(error_list)}， 缺失数据为{error_list}")


if __name__ == "__main__":
    binary_file_path = '/data/yy_data/five_minute_data/41f_5m_std_data_d/bin_data/688728.bin'
    a = read_record_from_binary(binary_file_path, 41, 0, record_count=1)
    pass
