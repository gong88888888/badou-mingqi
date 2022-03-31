# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

"""
@Project ：badou-mingqi 
@File    ：RNNTest.py
@Author  ：luigi
@Date    ：2022/3/31 5:07 PM 
"""

class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TorchRNN, self).__init__()
        self.layer = nn.RNN(input_size, hidden_size)
    
    def forward(self,x):
        return self.layer(x)
    
    
class DiyModel:
    def __init__(self, w_ih, w_hh, hidden_size):
        self.w_ih = w_ih     
        self.w_hh = w_hh
        self.hidden_size = hidden_size

    def forward(self,x):
        memory = np.zeros(self.hidden_size)

        # x 是input，其大小为m * n矩阵，m表示句子中的字数，n表示每个字所表示的向量长度
        # x_frame表示句子中的每个字，
        for x_frame in x:
            y = np.dot(self.w_ih, x_frame) + memory
            memory = np.tanh(np.dot(self.w_ih, x_frame))






