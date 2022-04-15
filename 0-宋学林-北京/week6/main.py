# -*- coding: utf-8 -*-
"""
    添加了训练结果写入CSV表格的函数 csv_writer(data)

"""
import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    # cuda_flag = False
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        # logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model.forward(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info("batch loss %f" % loss)
        # logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    # 保存模型：模型名字_lr=0.01_bs=64_pool=max_epoch=20__acc=80.00
    model_path = os.path.join(config["model_path"],
                              "%s__lr=%.3f__hs=%d__bs=%d__pool=%s__epoch=%d__acc=%.2f%%.pth" % (
                                  config["model_type"], config["learning_rate"],
                                  config["hidden_size"], config["batch_size"],
                                  config["pooling_style"], epoch, acc * 100))
    torch.save(model.state_dict(), model_path)  # 保存模型权重
    return acc


def csv_writer(data):
    """
    将结果写入csv文件
    """
    a = []
    dict = data[0]
    for headers in dict.keys():  # 把字典的键取出来
        a.append(headers)
    header = a  # 把列名给提取出来，用列表形式呈现
    # a表示以“追加”的形式写入,“w”的话，表示在写入之前会清空原文件中的数据
    # newline是数据之间不加空行
    with open('result.csv', 'a', newline='', encoding='utf-8') as f:
        # 提前预览列名，当下面代码写入数据时，会将其一一对应。
        writer = csv.DictWriter(f, fieldnames=header)
        # 写入列名
        writer.writeheader()
        # 写入数据
        writer.writerows(data)
    print("数据已经写入成功！！！")


if __name__ == "__main__":
    # main(Config)

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # # 超参数的网格搜索
    result_list = []
    for model in ["gated_cnn", "fast_text", "lstm", "rnn", "stack_gated_cnn", "rcnn"]:
        Config["model_type"] = model
        for lr in [1e-3, 5e-3]:
            Config["learning_rate"] = lr
            for hidden_size in [128, 256, 512]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["max", "avg"]:
                        Config["pooling_style"] = pooling_style
                        Config["acc"] = main(Config)  # 记录最后一轮的准确率
                        dict_temp = {"model_type": Config['model_type'],
                                     "max_length": Config['max_length'],
                                     "hidden_size": Config['hidden_size'],
                                     "kernel_size": Config['kernel_size'],
                                     "num_layers": Config['num_layers'],
                                     "epoch": Config['epoch'],
                                     "batch_size": Config['batch_size'],
                                     "pooling_style": Config['pooling_style'],
                                     "optimizer": Config['optimizer'],
                                     "learning_rate": Config['learning_rate'],
                                     "seed": Config['seed'],
                                     "acc": Config['acc']}
                        # print("当前配置：\n", Config)
                        result_list.append(dict_temp)
    csv_writer(result_list)