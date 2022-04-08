# -*- coding: utf-8 -*-
import json

import torch
import torch.nn as nn
import jieba

"""
@Project ：badou-mingqi 
@File    ：language_model.py
@Author  ：luigi
@Date    ：2022/4/6 3:50 PM 
"""

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, vector_dim, max_len, hidden_size):
        super(LanguageModel, self).__init__()
        self.word_vectors = nn.Embedding(vocab_size, vector_dim)
        self.inner_projection_layer = nn.Linear(vector_dim * max_len, hidden_size)
        self.outer_projection_layer = nn.Linear(hidden_size, hidden_size)
        self.x_projection_layer = nn.Linear(vector_dim * max_len, hidden_size)
        self.project_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, context):
        context_embedding = self.word_vectors(context) #output shape = batch_size, max_length , vector_dim

        #论文公式： y = b + Wx + U * tanh(d + Hx)， 其中x为每个词向量的拼接
        #词向量拼接
        x = context_embedding.view(context_embedding.shape[0], -1) # shape = batch_size, vector_dim * max_length

        # Hx +d
        inner_projection = self.inner_projection_layer(x) #shape = batch_size, hidden_size
        # tanh(d + Hx)
        inner_projection = torch.tanh(inner_projection) #shape = batch_size, hidden_size
        # U * tanh(d + Hx)
        outer_projection = self.outer_projection_layer(inner_projection) #shape = batch_size, hidden_size
        # Wx
        x_projection = self.x_projection_layer(x) #shape = batch_size, hidden_size
        # y = Wx + Utanh(hx+d) + b
        y = x_projection + outer_projection     #shape = batch_size, hidden_size
        #softmax后输出预测概率, 训练的目标是让y_pred对应到字表中某个字
        y_pred = torch.softmax(y, dim=-1)  #shape = batch_size, hidden_size
        return y_pred

def build_vocab():
    vocab = {}
    i=0
    with open('corpus.txt', 'r') as f:
        for line in f.readlines():
            seg_list = jieba.cut(line, cut_all=False)
            for word in seg_list:
                if vocab.get(word,None) is None:
                    i +=1
                vocab[word] = vocab.get(word,i)
    # 保存词表
    with open('vocab.json', 'w', encoding="utf-8") as f:
        f.write(json.dumps(vocab, ensure_ascii=False, indent=2))

def build_dataset():
    dataset_x = []
    dataset_y = []
    vocab = json.load(open('vocab.json', "r", encoding="utf-8"))  # 加载字典表
    with open('corpus.txt', 'r') as f:
        for line in f.readlines():
            seg_list = list(jieba.cut(line, cut_all=False))
            concat = ""
            for i, word in enumerate(seg_list):
                if i < len(seg_list) -1:
                    dataset_x.append(vocab[seg_list[i]])
                    dataset_y.append(vocab[seg_list[i+1]])

        return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


result = build_dataset()


