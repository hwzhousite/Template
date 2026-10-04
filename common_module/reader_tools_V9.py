"""
数据读取接口V9

每次返回wind小市值成分股的全部的股票数据

"""
import datetime
import json
import math
import os
import copy
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
    """
    返回交易日的列表，包含开头结尾
    """
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


def read_record_from_binary(binary_file_path, field_count, read_index, record_count=1, accuracy='f'):
    """
    从二进制文件中读取第read_index条记录开始的连续record_count条记录
    :param binary_file_path:  二进制文件路径
    :param field_count:       每条记录的字段数
    :param read_index:        要读取的第1条记录的索引下标（从0开始）
    :param record_count:      要读取的记录数
    :param accuracy:          精度，默认为f单精度，d为双精度
    :return: [[]]
    """
    byt = 8 if accuracy == 'd' else 4
    d_type = np.float64 if accuracy == 'd' else np.float32

    with open(binary_file_path, 'rb') as binary_file:
        # 跳转到第m条记录的起始位置
        binary_file.seek(field_count * read_index * byt, 0)  # 每个浮点数占用byt个字节
        data = binary_file.read(record_count * field_count * byt)
        record_set = np.frombuffer(data, dtype=d_type).reshape(record_count, field_count)
        return record_set


def read_record_from_binary_step(binary_file_path, field_count, read_index, record_count=1, interval=1, accuracy='f'):
    """
    从二进制文件中读取第read_index条记录开始的连续record_count条记录, 增加步长
    :param binary_file_path:  二进制文件路径
    :param field_count:       每条记录的字段数
    :param read_index:        要读取的第1条记录的索引下标（从0开始）
    :param record_count:      要读取的记录数
    :param interval:          取记录的间隔，即每隔interval条记录取一条
    :param accuracy:          精度，默认为f单精度，d为双精度
    :return: [[]]
    """
    byt = 8 if accuracy == 'd' else 4
    d_type = np.float64 if accuracy == 'd' else np.float32

    data_list = []
    with open(binary_file_path, 'rb') as binary_file:
        for i in range(0, record_count, interval):
            # 跳转到第m条记录的起始位置
            offset = (read_index + i) * field_count * byt
            binary_file.seek(offset, 0)  # 每个浮点数占用byt个字节
            data = binary_file.read(field_count * byt)
            if len(data) == field_count * byt:
                record = np.frombuffer(data, dtype=d_type)
                data_list.append(record)
    # 如果没有读取到任何记录，返回一个空的二维数组
    if not data_list:
        return np.empty((0, field_count), dtype=d_type)

    # 将读取到的记录列表转换为 numpy.ndarray
    result = np.vstack(data_list)
    return result


def read_one_stock_seq(data_path_bin, stock_code, start_idx, data_file_fields, need_seq_len, m5_first_only=False, accuracy='f'):
    """
    读取一只股票序列
    :param data_path_bin: 数据文件路径
    :param stock_code: 股票代码int
    :param start_idx: 开始索引
    :param data_file_fields: 数据文件字段数
    :param need_seq_len: 序列长度L
    :param m5_first_only: 仅仅返回第一个五分钟
    :param accuracy: 数据集精度
    :return: 股票序列[L,C]
    """

    # 股票代码转换为字符串，并在前面补零，使其长度为6
    # stock_code_str = str(stock_code).zfill(6)
    stock_code_str = stock_code
    file = os.path.join(data_path_bin, f"{stock_code_str}.bin")
    if m5_first_only:
        data_bin = read_record_from_binary_step(file, data_file_fields, start_idx, need_seq_len, 48, accuracy=accuracy)
    else:
        data_bin = read_record_from_binary(file, data_file_fields, start_idx, need_seq_len, accuracy=accuracy)
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


def get_n_step_trade_date(start_date, n, datefmt='%Y%m%d'):
    """
    如果n>0，返回n个交易日之后的交易日
    如果n<0，返回n个交易日之前的交易日
    """
    # 将字符串转换为datetime对象
    start = datetime.datetime.strptime(str(start_date), datefmt)

    # 当前日期从起始日期开始
    current_date = start

    # 循环直到当前日期超过结束日期
    while n:
        if n > 0:
            # 检查当前日期是否是周六或周日
            current_date += datetime.timedelta(days=1)
            if current_date.weekday() < 5:  # 0-4 是周一到周五
                n = n - 1
        else:
            current_date -= datetime.timedelta(days=1)
            if current_date.weekday() < 5:  # 0-4 是周一到周五
                n = n + 1

    date_int = int(current_date.strftime('%Y%m%d'))
    return date_int


class DatasetConfig:
    def __init__(self):
        # 数据集配置字典，存储多个数据集的配置信息
        self.config = {
        "tb_xt_5m_60f": {
            "data_path_bin": '/data/yy_data/five_minute_data/xt_tb_5min/bin_data',
            "index_file": '/data/yy_data/five_minute_data/xt_tb_5min/index.json',
            "scaler_file": '/data/yy_data/five_minute_data/xt_tb_5min/scaler_info.txt',
            "channels": 60,
            "period": 'five_minute',
            "accuracy": 'd',
            "raw_data": 'Y',
        },
        "ts_orig_102f": {
            "data_path_bin": '/data/yy_data/tushare_data/102f_250522/bin_data',
            "index_file": '/data/yy_data/tushare_data/102f_250522/index.json',
            "scaler_file": '/data/yy_data/tushare_data/102f_250522/scaler_info.txt',
            "channels": 102,
            "period": 'day',
            "accuracy": 'd',
            "raw_data": 'Y',
        },
        "ts_orig_104f": {
            "data_path_bin": '/data/yy_data/tushare_data/104f_250618/bin_data',
            "index_file": '/data/yy_data/tushare_data/104f_250618/index.json',
            "scaler_file": '/data/yy_data/tushare_data/104f_250618/scaler_info.txt',
            "channels": 104,
            "period": 'day',
            "accuracy": 'd',
            "raw_data": 'Y',
            "float_share_index": 30,
        },
        "ts_orig_106f": {
            "data_path_bin": '/data/yy_data/tushare_data/106f_250619/bin_data',
            "index_file": '/data/yy_data/tushare_data/106f_250619/index.json',
            "scaler_file": '/data/yy_data/tushare_data/106f_250619/scaler_info.txt',
            "channels": 106,
            "period": 'day',
            "accuracy": 'd',
            "raw_data": 'Y',
            "float_share_index": 30,
        },
        "wind_304f": {
                "data_path_bin": '/data/yy_data/wind/304f_20100104/bin_data',
                "index_file": '/data/yy_data/wind/304f_20100104/index.json',
                "scaler_file": '/data/yy_data/wind/304f_20100104/scaler_info.json',
                "channels": 304,
                "period": "day",
                "accuracy": "d",
                "raw_data": "Y",
                "float_share_index": 32,
            },
        "d12_402f": {
            "data_path_bin": "/data/yy_data/qd/train_XNGS/bin_data",
            "index_file": "/data/yy_data/qd/train_XNGS/index.json",
            "scaler_file": "/data/yy_data/qd/train_XNGS/scaler_info.json",
            "channels": 402,
            "period": 'day',
            "accuracy": 'd',
            "raw_data": 'Y',
        },
        "ts_orig_107f": {
            "data_path_bin": '/data/raw_generated_data/tushare_data/107f_wind_index/bin_data',
            "index_file": '/data/raw_generated_data/tushare_data/107f_wind_index/index.json',
            "scaler_file": '/data/raw_generated_data/tushare_data/107f_wind_index/scaler_info.txt',
            "channels": 107,
            "period": 'day',
            "accuracy": 'd',
            "raw_data": 'Y',
            "float_share_index": 30,
        },
        "ts_orig_110f": {
            "data_path_bin": '/data/yy_data/tushare_data/110f_infex_weight/bin_data',
            "index_file": '/data/yy_data/tushare_data/110f_infex_weight/index.json',
            "scaler_file": '/data/yy_data/tushare_data/110f_infex_weight/scaler_info.txt',
            "channels": 110,
            "period": 'day',
            "accuracy": 'd',
            "raw_data": 'Y',
            "float_share_index": 30,
        },
        "ts_orig_108f": {
            "data_path_bin": '/data/raw_generated_data/tushare_data/108f_infex_weight/bin_data',
            "index_file": '/data/raw_generated_data/tushare_data/108f_infex_weight/index.json',
            "scaler_file": '/data/raw_generated_data/tushare_data/108f_infex_weight/scaler_info.txt',
            "channels": 108,
            "period": 'day',
            "accuracy": 'd',
            "raw_data": 'Y',
            "float_share_index": 30,
        },
        "ts_orig_110f_zh": {
            "data_path_bin": '/data/yy_data/tushare_data/110f_index_weight_zh/bin_data',
            "index_file": '/data/yy_data/tushare_data/110f_index_weight_zh/index.json',
            "scaler_file": '/data/yy_data/tushare_data/110f_index_weight_zh/scaler_info.txt',
            "channels": 110,
            "period": 'day',
            "accuracy": 'd',
            "raw_data": 'Y',
            "float_share_index": 30,
        },
        "ts_orig_111f_zh": {
                "data_path_bin": '/data/yy_data/tushare_data/111f_index_data/bin_data',
                "index_file": '/data/yy_data/tushare_data/111f_index_data/index.json',
                "scaler_file": '/data/yy_data/tushare_data/111f_index_data/scaler_info.txt',
                "channels": 111,
                "period": 'day',
                "accuracy": 'd',
                "raw_data": 'Y',
                "float_share_index": 30,
            },
        }

    def get_config(self, dataset_name):
        """
        根据数据集名称获取配置，并自动验证文件和文件夹的有效性
        """
        if dataset_name not in self.config:
            raise ValueError(f"数据集 {dataset_name} 配置不存在")

        config = self.config[dataset_name]

        # 在获取配置时，自动校验文件和文件夹
        self._validate_file(config["index_file"])
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


class DiskBinDatasetTW(Dataset):
    # 从磁盘文件读取数据的Dataset
    def __init__(self, DSD, ds, transform=None):
        """
        初始化数据集
        :param args: 参数
        """
        self.DSD = DSD
        self.args = DSD.args
        self.ds = ds  # 数据集
        self.dict_stock = DSD.dict_stock  # 股票索引字典 {股票代码: {日期: 所在行序号}}
        self.ms_dict_stock = DSD.ms_dict_stock  # 股票索引字典 {股票代码: {日期: 所在行序号}}
        self.dict_date = DSD.dict_date  # 日期索引字典 {日期: {股票代码，所在行序号}}
        self.data_set_dict = DSD.data_set_dict  # {日期, [股票代码列表]}
        self.all_dates_index = DSD.all_dates_index  # 日期索引字典 {日期: 所在行序号}
        self.transform = transform  # 转换操作
        self.period = self.args.period  # 周期

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

        data = self.get_sample(day, group_idx)

        # 应用转换操作
        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_sample(self, day, group_idx):
        """
        获取一个样本的数据
        :param day 日期
        :param group_idx 组索引
        :return: input_seq[M,L,C], future_seq[M,P,C], global_seq[G,L,C]
        """

        si_stock_list = self.get_si_stock_list(day, group_idx)
        # print("SI股票列表：", si_stock_list)

        i_seq, ms_seq = self.read_si_seq(si_stock_list, day, self.args.day_time_step, self.args.minute_time_step,)  # [M,L,C]

        return i_seq, ms_seq

    def read_si_seq(self, stock_list, start_date_key,day_time_step, minute_time_step):
        """
        读取一组股票序列
        :param stock_list: 股票列表S
        :param start_date_key: 开始日期，日线数据的开始日期
        :param need_seq_len: 序列长度L
        :return: 股票序列[S,L,C]_
        """

        # 单线程读取
        seq = []
        ms_seq = []
        for stock_code in stock_list:
            stock_first_date = self.dict_stock[stock_code][0]
            si = self.all_dates_index[stock_first_date]  # 股票上市日期在全体日期列表中的索引
            ei = self.all_dates_index[start_date_key]  # 现在要读取的时序的首个日期在全体日期列表中的索引
            start_idx = ei - si  # 计算得到要读取的数据在该股票文件中的偏移位置

            # 分钟数据
            ms_stock_first_date = self.DSD.ms_dict_stock[stock_code][0]
            ms_si = self.all_dates_index[ms_stock_first_date]  # 股票上市日期在全体日期列表中的索引
            # 计算分钟数据的起始日期
            step = self.args.day_before_num - self.args.minute_before_num
            minute_start_date = get_n_step_trade_date(start_date_key, step)
            ms_ei = self.all_dates_index[minute_start_date]  # 现在要读取的时序的首个日期在全体日期列表中的索引
            ms_start_idx = ms_ei - ms_si  # 计算得到要读取的数据在该股票文件中的偏移位置
            assert ms_start_idx >= 0, f"分总汇总数据集起始索引必须大于0,{stock_code} ms_si-{ms_si} ms_ei-{ms_ei}"
            ms_start_idx = ms_start_idx * 48

            # 读取一只股票的数据
            stock_data = read_one_stock_seq(self.args.data_path_bin, stock_code, start_idx,
                                            self.args.channels,
                                            day_time_step, False, self.args.accuracy)  # [L,C]

            # 读取同一只股票的由分钟总结生成的数据
            ms_stock_data = read_one_stock_seq(self.DSD.ms_data_config['data_path_bin'], stock_code, ms_start_idx,
                                               self.DSD.ms_data_config['channels'],
                                               minute_time_step, self.args.m5_first_only, self.DSD.ms_data_config['accuracy'])  # [L,C]

            seq.append(stock_data)
            ms_seq.append(ms_stock_data)
        seq = torch.stack(seq, dim=0)  # [S,L,C]
        ms_seq = torch.stack(ms_seq, dim=0)  # [S,P,C]
        return seq, ms_seq

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
            return stock_list[-self.args.sicount:]
        else:
            return stock_list[start_idx:start_idx + self.args.sicount]


class StockDataReaderTW:
    """
    读取日线数据和分钟线数据的data_reader
    """
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.args.day_time_step = args.day_before_num + args.day_after_num
        self.args.minute_time_step = args.minute_before_num * 48 + args.minute_after_num * 48  # 分钟长度

        # 配置数据集的配置
        dc = DatasetConfig()
        data_config = dc.get_config(args.data_name)
        self.args.data_path_bin = data_config.get('data_path_bin')
        self.args.index_file = data_config.get('index_file')
        self.args.scaler_file = data_config.get('scaler_file')
        self.args.channels = data_config.get('channels')
        self.args.period = data_config.get('period')
        self.args.accuracy = data_config.get('accuracy')

        self.ms_data_config = dc.get_config(args.ms_data_name)

        self.dict_stock = {}  # 股票索引字典 {股票代码: {日期: 所在行序号}}
        self.dict_date = {}  # 日期索引字典 {日期: {股票代码，所在行序号}}
        self.data_set_dict = None  # {日期, [股票代码列表]}
        self.all_dates_list = []
        self.all_dates_index = {}
        self.need_day_len = max(args.day_before_num, args.minute_before_num) + max(args.day_after_num, args.minute_after_num)  # 日线的序列长度 和 分钟线的序列长度的最大并集
        self.need_minute_len = math.ceil(self.args.minute_time_step / 48) + 1  # 分钟线的序列天长度
        self.init()
        self.stride_offset = -1

    def init(self):
        # 加载索引文件
        print(f"{timestr()}初始化数据加载器。。。")
        time_start = time.time()
        self.load_index_from_file()
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

        with open(self.ms_data_config.get('index_file'), 'r', encoding='utf-8') as file:
            ms_dict_stock = json.load(file)

        # 剔除两个数据集不相交的股票代码
        dict_stock = {k: v for k, v in dict_stock.items() if k in ms_dict_stock}
        ms_dict_stock = {k: v for k, v in ms_dict_stock.items() if k in dict_stock}

        # 由于从json文件中加载字典时，所有的key都是字符串，因此需要将key转换为int类型
        # 在V2版本中，这里的value是一个list，只有两个元素，分别是首末日期int
        self.dict_stock = {int(k): v for k, v in dict_stock.items()}
        self.ms_dict_stock = {int(k): v for k, v in ms_dict_stock.items()}

        # 根据股票索引来创建日期索引，方便后续根据日期来查找股票
        self.dict_date = {}  # 日期索引字典 {日期: [股票代码，所在行序号]}
        # 汇聚所有的股票代码索引，得到日期索引
        print(f"{timestr()}创建日期索引。。。")
        count = 0

        # 遍历所有股票代码索引，得到全体数据集的最小日期和最大日期
        min_date = 99999999
        max_date = 0
        for stock_code, stock_index in self.dict_stock.items():
            # 遍历所有股票代码索引，得到股票代码stock_code和首末日期date
            # 生成首末日期之间的所有日期列表
            ms_stock_index = self.ms_dict_stock[stock_code]  # 获取辅助数据集的索引
            min_date = min(min_date, stock_index[0], ms_stock_index[0])
            max_date = max(max_date, stock_index[1], ms_stock_index[1])

        # 生成全部日期列表(!!!注意，这里的日期是有序的，而且不包含周末！！！)
        self.all_dates_list = make_trade_date_list(min_date, max_date, '%Y%m%d')
        # 初始化日期索引字典
        # 创建日期索引字典用于加快查询速度
        self.all_dates_index = {}
        idx = 0
        for date in self.all_dates_list:
            self.dict_date[date] = []
            self.all_dates_index[date] = idx
            idx += 1

        for stock_code, stock_index in self.dict_stock.items():
            # 遍历所有股票代码索引，得到股票代码stock_code和首末日期date
            # 取日线数据集合分钟数据集的交集，生成首末日期之间的所有日期列表
            ms_stock_index = self.ms_dict_stock[stock_code]  # 获取辅助数据集的索引
            s_index = max(stock_index[0], ms_stock_index[0])
            e_index = min(stock_index[1], ms_stock_index[1])
            si = self.all_dates_index[s_index]
            ei = self.all_dates_index[e_index]
            online_dates = self.all_dates_list[si:ei + 1]
            for day in online_dates:
                self.dict_date[day].append(stock_code)  # 设定该日期包含这支股票
            count += 1

        # self.dict_date的key是日期已经是排序过的，不需要再排序
        print(f"{timestr()}创建日期索引，DONE")


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
            if len(start_online_stocks) < self.args.sicount:
                continue
            # 取出最后一天在线的股票列表
            end_online_stocks = set(self.dict_date[end_day])
            if len(end_online_stocks) < self.args.sicount:
                continue
            # 取出首末两天同时都在线的股票列表
            online_stocks = start_online_stocks.intersection(end_online_stocks)
            if len(online_stocks) < self.args.sicount:
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

    def create_data_loader(self, start_time, end_time, refresh=False, stride_shift=True):
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
        si = 0
        if stride_shift:
            self.stride_offset = (self.stride_offset + 1) % self.args.stride
            si = self.stride_offset

        # 计算实际的开始时间
        actual_start_time = get_n_step_trade_date(start_time, self.need_day_len * -1 + 1)

        for day_index in range(si, len(data_set_keys), self.args.stride):

            day = data_set_keys[day_index]
            if day < actual_start_time:
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

        if not data_set:
            raise Exception("数据集为空")

        data_loader = DataLoader(
            DiskBinDatasetTW(self, data_set),
            batch_size=self.args.batch_size, shuffle=self.args.shuffle,
            num_workers=self.args.num_workers, prefetch_factor=4)

        print(f"{timestr()}创建data_loader完成，耗时{time.time() - time_start:.2f}秒")
        return data_loader


class DiskBinDatasetON(Dataset):
    # 从磁盘文件读取数据的Dataset
    def __init__(self, DSD, args, ds, dict_stock, dict_date, data_set_dict, all_dates_index, period, transform=None):
        """
        初始化数据集
        :param args: 参数
        """
        self.DSD = DSD
        self.args = args
        self.ds = ds  # 数据集
        self.dict_stock = dict_stock  # 股票索引字典 {股票代码: {日期: 所在行序号}}
        self.dict_date = dict_date  # 日期索引字典 {日期: {股票代码，所在行序号}}
        self.data_set_dict = data_set_dict  # {日期, [股票代码列表]}
        self.all_data_set_dict = DSD.all_data_set_dict  # {日期, [股票代码列表]}
        self.all_dates_index = all_dates_index  # 日期索引字典 {日期: 所在行序号}
        self.transform = transform  # 转换操作
        self.period = period  # 周期
        # 获取指定加权的列索引和常数D
        self.jq_column_index = getattr(args, 'jq_column_index', [])
        self.D = getattr(args, 'D', 1)

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
        day = self.ds[idx]

        data = self.get_si_sample(day)

        # 应用转换操作
        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_si_sample(self, day, minute_idx=None):
        """
        获取一个样本的数据
        :param day 日期
        :return:
        """

        si_stock_list = self.get_si_stock_list(day)
        # print("SI股票列表：", si_stock_list)

        if self.period == 'five_minute':
            need_seq_len = (self.args.minute_before_num + self.args.minute_after_num) * 48
        else:
            need_seq_len = (self.args.day_before_num + self.args.day_after_num)

        i_seq = self.read_si_seq(si_stock_list, day, minute_idx, need_seq_len)  # [M,L+P,C]

        if self.args.if_bj:
            bj = self.DSD.exclude_select_set_dict[day]
            bj_seq = self.read_si_seq(bj, day, minute_idx, need_seq_len)
            return i_seq, bj_seq

        return i_seq


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
                                            need_seq_len, accuracy=self.args.accuracy)  # [L,C]
            seq.append(stock_data)
        seq = torch.stack(seq, dim=0)  # [S,L,C]
        return seq

    def get_si_stock_list(self, day):
        """
        获取一个组的股票列表
        :param day: 起始日期
        :return: 股票列表S
        """

        # 根据日期找到对应的股票列表
        stock_list = self.data_set_dict[day]
        return stock_list


    def read_gbjq_seq(self, stock_list, start_date_key, minute_idx, need_seq_len):
        """
        股本加权
        读取某个时间范围内的全部股票序列
        :param stock_list: 股票列表
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
                                            need_seq_len, accuracy=self.args.accuracy)  # [L,C]
            seq.append(stock_data)

        seq = torch.stack(seq, dim=0)  # [S,L,C]

        # 对这个stock_data进行处理，让每一行的股票价格的字段*流通股本/常数D
        if not self.args.float_share_index:
            raise Exception(f'缺失float_share_index，无法使用股本加权')
        float_share_index = self.args.float_share_index
        float_share = seq[:, :, float_share_index:float_share_index+1]
        jq_value = seq[:, :, self.jq_column_index]
        new_cols = float_share * jq_value / self.D
        seq = seq.clone()
        seq[:, :, self.jq_column_index] = new_cols

        # 求均值
        seq = torch.sum(seq, dim=0) / len(stock_list)

        seq[:, 0] = 0
        seq[:, 1] = 0

        return seq.unsqueeze(0)

    def read_ggdq_seq(self, stock_list, start_date_key, minute_idx, need_seq_len):
        """
        个股等权
        读取某个时间范围内的全部股票序列
        :param stock_list: 股票列表
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
                                            need_seq_len, accuracy=self.args.accuracy)  # [L,C]

            # 对这个stock_data进行处理，让每一行的股票数据除以最后一行的股票数据
            last_data = stock_data[-1, self.jq_column_index]
            jq_value = stock_data[:, self.jq_column_index]
            new_cols = jq_value / last_data
            invalid_mask = torch.isnan(new_cols) | torch.isinf(new_cols)
            new_cols[invalid_mask] = 0.0
            stock_data = stock_data.clone()
            stock_data[:, self.jq_column_index] = new_cols
            seq.append(stock_data)

        seq = torch.stack(seq, dim=0)  # [S,L,C]
        # 求均值
        seq = torch.sum(seq, dim=0) / len(stock_list)

        seq[:, 0] = 0
        seq[:, 1] = 0

        return seq.unsqueeze(0)


def fill_trading_dates(a_data):
    """用前一天的值为非周六日补全日期"""

    # 转换日期字符串为datetime
    dates = {datetime.datetime.strptime(d, '%Y%m%d'): v for d, v in a_data.items()}

    # 获取最小和最大日期
    min_date = min(dates.keys())
    max_date = max(dates.keys())

    # 补全逻辑
    all_dates = {}
    last_valid_value = None
    current = min_date

    while current <= max_date:
        date_str = current.strftime('%Y%m%d')

        if current.weekday() < 5:  # 周一到周五
            if current in dates:
                last_valid_value = dates[current]  # 更新最新有效值
                all_dates[date_str] = last_valid_value
            elif last_valid_value is not None:
                all_dates[date_str] = last_valid_value  # 用最近的有效值补全

        current += datetime.timedelta(days=1)

    return all_dates


class StockDataReaderON:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        # 配置数据集的配置
        dc = DatasetConfig()
        data_config = dc.get_config(args.data_name)
        self.args.data_path_bin = data_config.get('data_path_bin')
        self.args.index_file = data_config.get('index_file')
        self.args.scaler_file = data_config.get('scaler_file')
        self.args.channels = data_config.get('channels')
        self.args.period = data_config.get('period')
        self.args.accuracy = data_config.get('accuracy')
        self.args.float_share_index = data_config.get('float_share_index')

        self.dict_stock = {}  # 股票索引字典 {股票代码: {日期: 所在行序号}}
        self.dict_date = {}  # 日期索引字典 {日期: {股票代码，所在行序号}}
        self.data_set_dict = None  # {日期, [股票代码列表]}
        self.all_dates_list = []
        self.all_dates_index = {}
        if self.args.period == 'five_minute':
            self.need_day_len = self.args.minute_before_num + self.args.minute_after_num
        elif self.args.period == 'day':
            self.need_day_len = self.args.day_before_num + self.args.day_after_num
        else:
            raise ValueError("period目前只支持five_minute和day")

        self.init()  # 初始化数据
        self.stride_offset = -1

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
        # dict_stock = {int(k): v for k, v in dict_stock.items()}
        dict_stock = {str(k): v for k, v in dict_stock.items()}

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
        # 为了记录时间交集下的所有同时在线的股票,当select_stock_code为空时，all_data_set_dict和data_set_dict一致
        self.all_data_set_dict = {}  # 保存时间交集下的所有股票
        self.select_data_set_dict = {}  # 保存股票交集下的所有股票
        self.exclude_select_set_dict = {}  # 补集

        # 遍历整个日期索引，看看哪一天作为起始日期时，是可以构造出完整的输入序列的
        # 生成的每个一样本索引是一个三元组[日期，[SI股票代码列表]，[SG股票代码列表]]
        date_key_list = list(self.dict_date.keys())  # 日期列表(这个日期已经是排序过的)

        date_key_len = len(date_key_list)

        with open(self.args.index_con_path, 'r', encoding='utf-8') as f:
            a_data = json.load(f)

        a_data = fill_trading_dates(a_data)

        new_daily_stocks = {int(k): [i[:6] for i in v] for k, v in a_data.items()}

        if date_key_len < self.need_day_len:
            print(f"数据量不足，无法生成序列")
            return None
        for di in range(date_key_len - self.need_day_len + 1):
            # 遍历所有的合法日期
            start_day = date_key_list[di]
            end_day = date_key_list[di + self.need_day_len - 1]
            # 取出第一天在线的股票列表
            start_online_stocks = set(self.dict_date[start_day])
            if len(start_online_stocks) < self.args.sicount:
                continue
            # 取出最后一天在线的股票列表
            end_online_stocks = set(self.dict_date[end_day])
            if len(end_online_stocks) < self.args.sicount:
                continue
            # 取出首末两天同时都在线的股票列表
            online_stocks = start_online_stocks.intersection(end_online_stocks)
            if len(online_stocks) < self.args.sicount:
                continue

            jiaoji = set()
            if end_day in new_daily_stocks:
                jiaoji = online_stocks.intersection(set(new_daily_stocks[end_day]))

            if not jiaoji:
                continue

            self.data_set_dict[start_day] = list(jiaoji)

            self.exclude_select_set_dict[start_day] = list(set(online_stocks) - jiaoji)


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

    def create_data_loader(self,start_time, end_time, refresh=False,stride_shift=True):
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

        # 计算实际的开始时间
        actual_start_time = get_n_step_trade_date(start_time, self.need_day_len * -1 + 1)

        if self.args.period == 'day':
            si =0
            if stride_shift:
                self.stride_offset = (self.stride_offset + 1) % self.args.stride
                si = self.stride_offset
            for day_index in range(si, len(data_set_keys), self.args.stride):

                day = data_set_keys[day_index]
                if day < actual_start_time:
                    continue

                di = self.all_dates_list.index(day)  # 算出该日期在全体日期列表中的索引
                actual_end_day = self.all_dates_list[di + self.need_day_len - 1]  # 获取时间窗口实际结束的天
                if actual_end_day > end_time:
                    continue

                data_set.append(day)

        elif self.args.period == 'five_minute':
            for minute_index in range(0, len(data_set_keys) * 48, self.args.stride):
                day_index = minute_index // 48  # 算出天的索引
                day = data_set_keys[day_index]
                if day < actual_start_time:
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
            DiskBinDatasetON(self, self.args, data_set, self.dict_stock, self.dict_date, self.data_set_dict,
                           self.all_dates_index, self.args.period),
            batch_size=self.args.batch_size, shuffle=self.args.shuffle,
            num_workers=self.args.num_workers, prefetch_factor=4)

        print(f"{timestr()}创建data_loader完成，耗时{time.time() - time_start:.2f}秒")
        return data_loader

class DataReader:
    @staticmethod
    def create_data_reader(args):
        dc = DatasetConfig()
        if args.data_name in dc.config and args.ms_data_name in dc.config:
            # 日线数据和分钟数据
            return StockDataReaderTW(args)

        elif args.data_name in dc.config:
            # 日线数据
            return StockDataReaderON(args)

        elif args.ms_data_name in dc.config:
            # 分钟数据
            m_args = copy.deepcopy(args)
            m_args.data_name = args.ms_data_name
            return StockDataReaderON(m_args)

        else:
            raise ValueError('data_name 和 ms_data_name 至少有一个合法选项')


class CheckData:
    def __init__(self, **kwargs):
        self.args = copy.deepcopy(kwargs.get("args"))
        self.scaler_info = {}
        # 配置数据集的配置
        dc = DatasetConfig()
        data_config = dc.get_config(self.args.data_name)
        self.data_path_bin = data_config.get('data_path_bin')
        self.index_file = data_config.get('index_file')
        self.scaler_file = data_config.get('scaler_file')
        self.channels = data_config.get('channels')
        self.period = data_config.get('period')

        with open(self.scaler_file, 'r', encoding='utf-8') as f:
            scaler_info = json.load(f)
        self.scaler_info = scaler_info
        self.field_list = list(self.scaler_info.keys())

        torch.set_printoptions(sci_mode=False)

    # 还原原始数据
    def restore_original_data(self, batch_data, n=3):
        """
        验证数据：将归一化后的数据还原为原始数据
        :param n: 用于打印第一个数据样本的前n个时间步
        :param batch_data:
        :return:
        """

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
                if  tensor_seq[b, s, 0, 0] != tensor_seq[b, s, -1, 0]:
                    raise Exception(f"时间维度上股票编码不一致: b:{b} s:{s} {tensor_seq[b, s, 0, 0]} {tensor_seq[b, s, -1, 0]}")

                year_index = self.field_list.index('gen_year')
                month_index = self.field_list.index('gen_month')
                day_index = self.field_list.index('gen_day')

                start_year = tensor_seq[b, s, 0, year_index]
                end_year = tensor_seq[b, s, -1, year_index]

                start_month = tensor_seq[b, s, 0, month_index]
                end_month = tensor_seq[b, s, -1, month_index]

                start_day = tensor_seq[b, s, 0, day_index]
                end_day = tensor_seq[b, s, -1, day_index]

                if self.period == 'five_minute':
                    minute_index = self.field_list.index('gen_minute')

                    start_minute = tensor_seq[b, s, 0, minute_index]
                    end_minute = tensor_seq[b, s, -1, minute_index]

                    start_date_time = int(
                        f"{int(start_year):04d}{int(start_month):02d}{int(start_day):02d}{int(start_minute):02d}")
                    end_date_time = int(
                        f"{int(end_year):04d}{int(end_month):02d}{int(end_day):02d}{int(end_minute):02d}")
                    assert (start_time * 100 + 1) <= start_date_time and (end_time * 100 + 48) >= end_date_time, f"存在信息泄露: start_date_time:{start_date_time}, end_date_time:{end_date_time}"
                else:
                    start_date_time = int(f"{int(start_year):04d}{int(start_month):02d}{int(start_day):02d}")
                    end_date_time = int(f"{int(end_year):04d}{int(end_month):02d}{int(end_day):02d}")
                    assert start_time <= start_date_time and end_time >= end_date_time, f"存在信息泄露: start_date_time:{start_date_time}, end_date_time:{end_date_time}"

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
        :param kwargs: input_seq

        1. 检验day_seq的形状是否正确
        2. 检验时间维度上股票编码是否一致
        3. 检验股票维度上的时间是否一致
        4. 检验时间是否存在起始时间和结束时间范围内，防止信息泄露

        """
        print("开始验证数据。。。")
        # 检查每个batch的形状是否正确

        input_seq = args[0]

        start_time = kwargs.get("start_time", 0)
        end_time = kwargs.get("end_time", 99999999)

        look_back_windows = kwargs.get("look_back_windows")

        start_time = get_n_step_trade_date(start_time, look_back_windows * -1 + 1)

        if isinstance(input_seq, torch.Tensor):
            c_day_seq_shape = [self.args.batch_size, self.args.sicount, look_back_windows, self.channels]
            day_seq_shape = list(input_seq.shape)
            if c_day_seq_shape != day_seq_shape:
                print(f"day_seq形状有误,{c_day_seq_shape} {day_seq_shape}")

        # 验证数据的细节
        self.validate_detail(input_seq, start_time, end_time)

        print("验证数据完成")

    def validation_data_minute(self, *args, **kwargs):
        """
        验证分钟数据
        :param kwargs: input_seq

        1. 检验minute_seq的形状是否正确
        2. 检验时间维度上股票编码是否一致
        3. 检验股票维度上的时间是否一致
        4. 检验时间是否存在起始时间和结束时间范围内，防止信息泄露

        """
        print("开始验证数据。。。")
        # 检查每个batch的形状是否正确

        input_seq = args[0]

        start_time = kwargs.get("start_time", 0)
        end_time = kwargs.get("end_time", 99999999)

        look_back_windows = kwargs.get("look_back_windows", 0)
        m5_first_only = kwargs.get("m5_first_only")

        if not m5_first_only:
            look_back_windows *= 48

        if isinstance(input_seq, torch.Tensor):
            c_day_seq_shape = [self.args.batch_size, self.args.sicount, look_back_windows, self.channels]
            day_seq_shape = list(input_seq.shape)
            if c_day_seq_shape != day_seq_shape:
                print(f"minute_seq形状有误,{c_day_seq_shape} {day_seq_shape}")

        # 验证数据的细节
        self.validate_detail(input_seq, start_time, end_time)

        print("验证数据完成")




if __name__ == "__main__":
    # from argparse import Namespace
    # from torch import Tensor
    # args = Namespace()
    # args.data_name = 'ts_day_54f'
    # cd = CheckData(args=args)
    # print("全天的数据：")
    # min_data = Tensor([[[[301330, 2.8490e+03,  3.0000e-01,  1.5000e-01,  1.6500e-01,  8.5000e-01,
    #      6.6667e-01,  5.1613e-01,  1.6667e-01,  0.0000e+00,  7.1819e-01,
    #      7.1819e-01,  7.1819e-01,  7.1819e-01,  7.1819e-01, -6.6462e-03,
    #     -1.9944e-02, -3.4479e-01, -3.5523e-01, -6.0801e-01, -5.2191e-01,
    #     -1.8762e-01, -2.1260e-01, -1.5926e-02, -4.5453e-02, -7.7082e-03,
    #     -2.8025e-03, -5.2275e-01, -1.5285e-01, -1.2671e-01, -1.2994e-01,
    #     -2.9102e-01, -1.9143e-01, -1.9370e-01, -3.9509e-01, -4.4728e-01,
    #     -3.9138e-01, -4.4747e-01, -3.7788e-01, -3.8775e-01, -3.8010e-01,
    #     -3.9025e-01, -3.0288e-01, -3.0179e-01, -3.1288e-01, -3.1184e-01,
    #     -1.4295e-01, -1.5354e-01, -1.5271e-01, -1.5981e-01,  7.5887e-02,
    #      7.4169e-02,  4.0000e-01,  1.0000e+00]]]])
    #
    #
    # original_min = cd.restore_original_data(min_data)
    # print("分钟合成的数据：")
    # d_data = Tensor([[[[600567, 3.4750e+03,  1.0000e-01,  5.0000e-02,  6.5000e-01,  3.3333e-01,
    #      5.8065e-01,  5.0000e-01,  0.0000e+00,  2.8000e+00,  2.8400e+00,
    #      2.7800e+00,  2.8000e+00,  2.8000e+00, -5.4619e-03, -1.8227e-02,
    #     -7.7893e-02, -3.1347e-01, -6.3074e-02, -4.9569e-01, -1.4494e-01,
    #     -2.6871e-02, -9.3450e-02,  8.5172e-01, -4.9594e-01, -8.0919e-02,
    #     -1.7357e-01,  1.1844e-01,  6.7944e-01, -1.6199e-01, -3.6678e-01,
    #     -6.6958e-01, -6.0555e-01, -1.4956e-01, -4.1754e-01, -6.3604e-01,
    #     -6.9342e-01,  5.0000e-02,  0.0000e+00]]]])
    #
    # original_data = cd.restore_original_data(d_data)
    pass
