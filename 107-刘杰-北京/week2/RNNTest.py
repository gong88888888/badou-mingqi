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
        self.layer = nn.RNN(input_size, hidden_size, bias=False, batch_first=True)
    
    def forward(self, x):
        return self.layer(x)
    
    
class DiyModel:
    def __init__(self, w_ih, w_hh, hidden_size):
        self.w_ih = w_ih
        self.w_hh = w_hh
        self.hidden_size = hidden_size

    def forward(self,x):

        memory = np.zeros(self.hidden_size)
        output = []
        # 公式：ht = tanh(b + Whs + Ux)    其中s = t -1
        # x 是input，其大小为m * n矩阵，m表示句子中的字数，n表示每个字所表示的向量长度
        # x_frame表示句子中的每个字，
        for x_frame in x:
            ux = np.dot(self.w_ih, x_frame)
            wh = np.dot(self.w_hh, memory)
            memory = np.tanh(ux + wh)
            output.append(memory)
        return np.array(output), memory

def main():
    x = np.arange(9).reshape((3, 3))
    torch_x = torch.FloatTensor([x])
    hidden_size = 4
    torch_model = TorchRNN(torch_x.shape[-1], hidden_size)
    # print(torch_model.state_dict())
    w_ih = torch_model.state_dict()["layer.weight_ih_l0"]
    w_hh = torch_model.state_dict()["layer.weight_hh_l0"]
    print("w_ih shape", w_ih.shape)
    print("w_hh shape", w_hh.shape)

    output, memory = torch_model.forward(torch_x)
    print(output,"torch模型预测结果")
    print(memory, "torch模型预测隐含记忆单元结果")
    print("---------------")
    diyModel = DiyModel(w_ih, w_hh, hidden_size)
    output, memory = diyModel.forward(x)
    print(output,"diy模型预测结果")
    print(memory,"diy模型预测隐含记忆单元结果")
if __name__ == '__main__':
    main()







