# -*- coding: utf-8 -*-

import jieba
import gensim
from gensim.models import Word2Vec
import json
import numpy as np
import logging

"""
@Project ：badou-mingqi 
@File    ：word2vec_train.py
@Author  ：luigi
@Date    ：2022/4/8 2:53 PM 
"""

#训练模型
#corpus: [["cat", "say", "meow"], ["dog", "say", "woof"]]
#dim指定词向量的维度，如100
def train_w2v(corpus, dim):
    model = Word2Vec(corpus, vector_size=dim, min_count=3, sg=0)
    model.save("model.w2v")
    return model

#输入模型文件路径
#加载训练好的模型
def load_w2v_model(path):
    model = Word2Vec.load(path)
    return model


def main():
    sentences = []
    with open('title_20220408.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            seglist = jieba.lcut(line)
            sentences.append(seglist)
    model = train_w2v(sentences, 200)
    return

if __name__ == '__main__':
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # main()

    model = load_w2v_model('model.w2v')
    while True:
        string = input("input:")
        try:
            print(model.wv.most_similar(string))
        except:
            print("输入词不存在")
