# -*- coding: utf-8 -*-
from sklearn.model_selection import ParameterGrid           #网格搜索法
import openpyxl
import torch
torch.cuda.current_device()
import random
import os
import numpy as np
import pandas as pd
import logging
from logging import handlers
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]                   #默认显示warning级别及以上的的日志
# logger.debug('debug级别，一般用来打印一些调试信息，级别最低')
# logger.info('info级别，一般用来打印一些正常的操作信息')
# logger.warning('waring级别，一般用来打印警告信息')
# logger.error('error级别，一般用来打印一些错误信息')
# logger.critical('critical级别，一般用来打印一些致命的错误信息，等级最高')

# logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     filename='run_process.log', filemode='a')   #控制台不输出，仅写入文件
# logger = logging.getLogger(__name__)           #获取logger对象

# 创建一个logger并设置日志等级
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)                 #告诉logger要记录哪些级别的日志
#logger 的级别是先过滤的，所以被 logger 过滤的日志 handler 也是无法记录的，这样就可以只改 logger 的级别而影响所有输出。

# 定义日志文件
logFile = 'run_process.log'

# 定义Handler的日志输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s')

# 创建一个FileHandler,并将日志写入指定的日志文件中
fileHandler = logging.FileHandler(filename=logFile, mode='a', encoding='utf-8')   #追加写的方式，'w'覆盖之前的日志
fileHandler.setLevel(logging.INFO)             #告诉Handler要记录哪些级别的日志
fileHandler.setFormatter(formatter)


# 创建一个StreamHandler,将日志输出到控制台
streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.INFO)
streamHandler.setFormatter(formatter)

# handlers.RotatingFileHandler -> 按照大小自动分割日志文件，一旦达到指定的大小重新生成文件
# handlers.TimedRotatingFileHandler -> 按照时间自动分割日志文件

# 定义日志滚动条件，这里按日期-天保留日志
# timedRotatingFileHandler = handlers.TimedRotatingFileHandler(filename=logFile, when='D', encoding='utf-8')
# timedRotatingFileHandler.setLevel(logging.INFO)
# timedRotatingFileHandler.setFormatter(formatter)

# 添加Handler
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)
#logger.addHandler(timedRotatingFileHandler)

"""
模型训练主程序
"""

#设置随机种子
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):              #判断某一路径是否为目录，若目录不存在则创建
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)      #调用函数，加载训练数据
    #加载模型
    model = TorchModel(config)                                     #使用config.py中的配置信息初始化模型
    # 标识是否使用gpu，若可用把模型迁移到显存中
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)                    #按照config信息为模型配置优化器
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []                     #每个epoch清空上一次的损失
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            try:
                loss = model(input_ids, labels)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                        loss = model(input_ids, labels)
                else:
                    raise exception
            #loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())   #把每个batch的损失添加到列表中
            if index % int(len(train_data) / 2) == 0:        #每个epoch训练一半的batch后，显示当前损失
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))      #每个epoch的平均损失
        acc = evaluator.eval(epoch)                                      #调用评估函数，评估模型
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)     #模型保存路径
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

if __name__ == "__main__":
    # main(Config)                    #使用Config信息调用main函数
    # for model in ["cnn"]:           #循环不同的模型结构，输出不同模型的准确率
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])
    '''
    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    for model in ["gated_cnn"]:                             #模型列表
        Config["model_type"] = model
        for lr in [1e-3]:                                   #学习率列表
            Config["learning_rate"] = lr
            for hidden_size in [128]:                       #隐层列表
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:                #batch_size列表
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg"]:           #池化方式列表
                        Config["pooling_style"] = pooling_style
                        print("最后一轮准确率：", main(Config), "当前配置：", Config)      #输出每个模型在各种配置下的准确率
    '''
    model_type = ["fast_text", "lstm", "gru", "rnn", "cnn", "gated_cnn", "stack_gated_cnn",
                 "rcnn", "bert", "bert_lstm", "bert_cnn", "bert_mid_layer"]
    learning_rate = [1e-3]
    hidden_size = [64, 128, 256]
    batch_size = [32, 64]
    pooling_style = ["max", "avg"]
    optimizer = ["adam", "sgd"]
    param_grid = dict(model_type=model_type, learning_rate=learning_rate, hidden_size=hidden_size,
                      batch_size=batch_size, pooling_style=pooling_style, optimizer=optimizer)
    model_params =list(ParameterGrid(param_grid))
    statistics = []
    for i in range(len(model_params)):
        model_param = model_params[i]
        current_param = dict(Config, **model_param)
        statistic = dict(current_param, **dict(acc=main(current_param)))
        statistics.append(statistic)
    statistics = pd.DataFrame.from_dict(statistics)
    #寻找acc最大的那行参数
    best_acc = statistics['acc'].max()
    best_params = statistics.iloc[statistics['acc'].argmax()]
    print('最高正确率：', best_acc)
    best_params = best_params.loc[["model_type", "max_length", "hidden_size", "kernel_size", "num_layers", "batch_size",
                                   "pooling_style", "optimizer", "learning_rate"]]
    print('最优参数：\n', best_params)
    writer = pd.ExcelWriter('统计.xlsx')  # 写入Excel文件
    statistics.to_excel(writer)
    writer.save()
    writer.close()