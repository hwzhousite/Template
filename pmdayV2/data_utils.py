import torch
import os
import json
import copy
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils_date as DT


class DataNormalizer_SWSJ(torch.nn.Module):
    # 20250530 创建,用于对获取的一个batch数据进行标准化,
    # 之所以要单独写一个类,是因为自这个时段开始,所有的数据将以原始值的形式提供,
    # 由模型使用者自行决定是否要对这些数据进行标准化再输入到模型中.
    # 如果你的模型同时用到了多个数据集(如:同时用到了天数据和分钟数据),则需要创建多个DataNormalizer实例
    def __init__(self, scaler_file, priceFieldList):
        # 初始化时,请指定数据集对应的scaler_info文件的路径
        super(DataNormalizer_SWSJ, self).__init__()
        self.scaler_file = scaler_file
        assert os.path.exists(self.scaler_file), f"scaler_file not found: {self.scaler_file}"
        # 从文件中加载scaler信息字典
        self.scaler = json.load(open(scaler_file, 'r'))
        self.src_field_list = list(self.scaler.keys())  # 原始数据的字段名列表,即data_reader返回的一个batch数据的字段名
        self.output_field_list = copy.deepcopy(self.src_field_list)  # 输出数据的字段名列表默认与原始数据的字段名相同,即全部输出

        # 示例: 一般情况下,我们不需要首字段股票代码:ts_code, 可以将它移除
        # 也可以在初始化对象完成后, 调用对象的set_output_field_list() 方法设置输出数据的字段名列表
        # 注意设置字段名称时的大小写, 确保大小写一致,下同
        firstField = self.src_field_list[0]
        self.output_field_list.remove(firstField)

        if "311f" in scaler_file:
            self.year_name = 'f_d_year'
            self.month_name = 'f_d_month'
            self.day_name = 'f_d_day'
            self.week_name = 'f_d_week'
        if "395f" in scaler_file:
            self.year_name = 'f_d_year'
            self.month_name = 'f_d_month'
            self.day_name = 'f_d_day'
            self.week_name = 'f_d_week'
        else:
            self.year_name = 'gen_year'
            self.month_name = 'gen_month'
            self.day_name = 'gen_day'
            self.week_name = 'gen_week'

        # 示例: 价格类的字段,有可能需要保持原始值,即不进行标准化, 所以可以将它们添加到keep_original_list中
        # 也可以在初始化对象完成后, 调用对象的set_original_list() 方法设置不需要标准化的字段名列表
        # 本例中: 将高/开/低/收/前收/前5分钟均价/9点35分价格,共7个字段设置为保持原始值,不需要标准化
        self.keep_original_list = copy.deepcopy(priceFieldList)  # 保持原始值,不需要标准化的字段名列表
        self.y_idx = self.output_field_list.index(self.year_name)
        self.m_idx = self.output_field_list.index(self.month_name)
        self.d_idx = self.output_field_list.index(self.day_name)
        self.w_idx = self.output_field_list.index(self.week_name)

        self.normalize_info = None
        self.output_field_index = []  # 输出字段在原始字段中的索引表,用于快速提取数据
        for fn in self.output_field_list:
            fnidx = self.src_field_list.index(fn)
            self.output_field_index.append(fnidx)
        self.updateInfo()

        # print("*********数据集信息*********")
        # print(f"原始数据字段数:{len(self.src_field_list)}---{self.src_field_list}")
        # print(f"输出数据字段数:{len(self.output_field_list)}---{self.output_field_list}")
        # print(f"保持原始值字段数:{len(self.keep_original_list)}---{self.keep_original_list}")

    def get_output_channel_num(self):
        # 返回输入数据的通道数
        return len(self.output_field_list)

    def updateInfo(self):
        # 初始化normalize_info字典,用于进行标准化
        # 无论某个字段是否在最后需要输出,我们都计算出它的标准化数据,并记录在normalize_info张量中
        normalize_info = np.zeros((len(self.src_field_list), 2))
        for i, field_name in enumerate(self.src_field_list):
            field_info = self.scaler[field_name]
            # 字段类型名只有三种: NONE,ENUM,MMS
            if field_name in ['news_micro_score','news_macro_score']:
                # 对于news_micro_score和news_macro_score字段,取值范围为[-3,3],这里将其归一化到[-1,1]之间,只要除以3即可
                normalize_info[i] = np.array([0, 3])
            elif field_info[0] == 'NONE' or field_name in self.keep_original_list:
                # 对于 NONE 类型的字段,不需要进行标准化, 所以标准化信息设置为 [0, 1]
                # 如果用户指定某些字段需要保持原始值,则相当于NONE类型字段
                normalize_info[i] = np.array([0, 1])
            elif field_info[0] == 'ENUM':
                # 对于 ENUM 类型的字段,需要进行标准化, 将原始值压缩到0~1之间即可,
                # 所以标准化信息设置为 [min, max-min+1]
                normalize_info[i] = np.array([field_info[1], field_info[2] - field_info[1] + 1])
            else:
                # 对于 MMS 类型的字段,需要进行标准化, 所以标准化信息设置为 [mean, var]
                normalize_info[i] = np.array([field_info[3], field_info[4]])
        self.normalize_info = torch.from_numpy(normalize_info).float()

    def set_output_field_list(self, output_field_list):
        # 设置输出数据的字段名列表
        self.output_field_list = copy.deepcopy(output_field_list)
        self.output_field_index = []  # 输出字段在原始字段中的索引表,用于快速提取数据
        for fn in self.output_field_list:
            fnidx = self.src_field_list.index(fn)
            self.output_field_index.append(fnidx)

    def set_original_list(self, keep_original_list):
        # 设置不需要标准化的字段名列表,即这些字段将不会被标准化
        self.keep_original_list = copy.deepcopy(keep_original_list)
        self.updateInfo()

    def forward(self, data_batch, normalize=True):
        """
        输入一个batch数据,返回标准化后的batch数据
        :param data_batch: 一个batch的数据 [B,M,L,src_C]
        :param normalize: 是否进行标准化,默认True
        :return: 一个batch的数据 [B,M,L,ouput_C]
        """

        # 首先,传入本方法的data_batch是由data_reader从数据文件中读取的原始数据,包含该数据据的全部字段
        # 其形状为 [B,M,L,src_C],其中B为batch_size,M为股票数,L为时间步数,src_C为原始数据的特征数

        if normalize:
            # self.normalize_info是一个 [src_C, 2] 的矩阵,记录每个字段的标准化信息(均值,方差)
            if self.normalize_info.device != data_batch.device:
                self.normalize_info = self.normalize_info.to(data_batch.device)

            # 将原始数据中的全部字段都执行标准化(方法为: 减去均值除以方差)
            # 如果是枚举类型或者是NONE类型,或者是用户指定了不需要进行标准化的字段, 我们在前面生成
            # normalize_info时就已经将其设置为 [0, 1]了, 所以即便执行了减0除1的操作,也不会改变数据
            data_batch = (data_batch - self.normalize_info[:, 0]) / self.normalize_info[:, 1]

        # 截取需要输出的字段,并返回
        output_batch = data_batch[:, :, :, self.output_field_index]
        return output_batch

    def un_normalize(self, data_batch):
        """
        对一个batch数据进行反标准化,即将标准化后的数据还原到原始值
        :param data_batch: 一个batch的数据 [B,M,L,ouput_C]
        :return: 一个batch的数据 [B,M,L,ouput_C]
        """

        # 首先,传入本方法的data_batch是由forward方法标准化后的数据,仅包含self.output_field_list所指定的字段
        # 对于缺失的字段,我们没有办法获取,所以输出的字段数只能仍然是self.output_field_list所指定的字段数
        # 该方法只用于debug程序时用于查看原始数据,用到循环,速度慢
        B, M, L, C = data_batch.shape
        for i in range(C):
            # 对于每个字段,先获取该字段的标准化信息
            field_name = self.output_field_list[i]
            field_info = self.scaler[field_name]
            src_field_index = self.output_field_index[i]
            norm_info = self.normalize_info[src_field_index]
            # 字段类型名只有三种: NONE,ENUM,MMS
            if field_info[0] == 'NONE' or field_name in self.keep_original_list:
                # 对于 NONE 类型的字段,不需要进行反标准化, 所以直接返回
                continue
            elif field_info[0] == 'ENUM':
                # 对于 ENUM 类型的字段,需要进行反标准化, 将0~1之间的数值还原到原始值
                # 浮点数转换为整数时,需要就近取整,因此需要用round函数
                data_batch[:, :, :, i] = round(data_batch[:, :, :, i] * norm_info[1] + norm_info[0])
            else:
                # 对于 MMS 类型的字段,需要进行反标准化
                data_batch[:, :, :, i] = data_batch[:, :, :, i] * norm_info[1] + norm_info[0]
        return data_batch

    def make_futuer_seq(self, x, days, knownChannels, offset=1):
        """
        根据已知的序列数据制作未来序列
        :param x: 输入序列 [B, M, L, C]  (!!!注意x是已经经过标准化之后的数据!!!)
        :param days: 需要构造的未来天数
        :param knownChannels: 已知的通道数
        :param offset: 未来序列的偏移量,默认1
        :return: 未来序列 [B, M, days, C]
        """

        B, M, L, C = x.shape
        future = x[:, :, -1:, :].clone()  # [B, M, 1, C]
        future = future.repeat(1, 1, days, 1)  # [B, M, days, C]
        future[:, :, :, knownChannels:] = 0.  # 除了已知的通道，其他通道设置为0

        field_info = self.scaler[self.year_name]
        Y_min = field_info[1]
        Y_width = float(field_info[2] - field_info[1] + 1)
        field_info = self.scaler[self.month_name]
        M_min = field_info[1]
        M_width = float(field_info[2] - field_info[1] + 1)
        field_info = self.scaler[self.day_name]
        D_min = field_info[1]
        D_width = float(field_info[2] - field_info[1] + 1)
        field_info = self.scaler[self.week_name]
        W_min = field_info[1]
        W_width = float(field_info[2] - field_info[1] + 1)

        for b in range(B):
            last_date = self.make_date_from_tensor(x[b, 0, -1])  # 这是回看窗口最后一天的日期
            if offset > 0:
                last_date = DT.get_next_trade_day(str(last_date), offset) # 向前一天, 得到今天(当前交易日)的日期
            for i in range(days):
                last_date = DT.get_next_trade_day(str(last_date), 1)
                y, m, d, w = DT.parse_date(last_date)
                future[b, :, i, self.y_idx] = (y - Y_min) / Y_width  # 年份字段归一化
                future[b, :, i, self.m_idx] = (m - M_min) / M_width  # 月份字段归一化
                future[b, :, i, self.d_idx] = (d - D_min) / D_width  # 日期字段归一化
                future[b, :, i, self.w_idx] = (w - W_min) / W_width  # 星期字段归一化
        return future

    def get_batch_date(self, batchData):
        """
        获取日期
        :param batchData: [B,M,C]
        """

        field_info = self.scaler[self.year_name]
        fmin = field_info[1]
        fwidth = field_info[2] - field_info[1] + 1
        by = torch.round(batchData[:,:,self.y_idx] * fwidth + fmin)
        field_info = self.scaler[self.month_name]
        fmin = field_info[1]
        fwidth = field_info[2] - field_info[1] + 1
        bm = torch.round(batchData[:,:,self.m_idx] * fwidth + fmin)
        field_info = self.scaler[self.day_name]
        fmin = field_info[1]
        fwidth = field_info[2] - field_info[1] + 1
        bd = torch.round(batchData[:,:,self.d_idx] * fwidth + fmin)
        r = (by * 10000).long() + (bm * 100).long() + bd.long()
        return r

    def make_date_from_tensor(self,ts):
        # 将一个tensor中的时间戳转换为日期字符串
        # 输入: ts: [output_C]

        field_info = self.scaler[self.year_name]
        fmin=field_info[1]
        fwidth = field_info[2] - field_info[1] + 1
        year = round(ts[self.y_idx].item() * fwidth + fmin)
        field_info = self.scaler[self.month_name]
        fmin=field_info[1]
        fwidth = field_info[2] - field_info[1] + 1
        month = round(ts[self.m_idx].item() * fwidth + fmin)
        field_info = self.scaler[self.day_name]
        fmin=field_info[1]
        fwidth = field_info[2] - field_info[1] + 1
        day = round(ts[self.d_idx].item() * fwidth + fmin)
        return int(year * 10000) + int(month * 100) + int(day)



