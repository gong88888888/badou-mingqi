# -*- coding: utf-8 -*-
import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json

"""
@Project ：badou-mingqi 
@File    ：NLPTask.py
@Author  ：luigi
@Date    ：2022/4/1 11:35 AM 
"""

class TorchModel(nn.Module):
    def __init__(self, sentence_length, vector_dim, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        self.pool = nn.AvgPool1d(sentence_length)
        self.rnn = nn.RNN(vector_dim, 10, bias=False, batch_first=True)
        self.liner = nn.Linear(10, 1)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None):
        x = self.embedding(x)                           #x为输入语句对应的字典索引 (batch_size * sen_len) -> (batch_size * sen_len * vector_dim)
        x = self.rnn(x)[0]                              # (batch_size * sen_len * vector_dim) -> (batch_size * sen_len * 10)
        x = self.pool(x.transpose(1,2)).squeeze()       # (batch_size * sen_len * 10) -> (batch_size * 10)
        x = self.liner(x)                               # (batch_size * 10) -> (batch_size * 1)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y, y_pred)
        else:
            return y_pred

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    # chars = "abcdefghijklmnopqrstuvwxyz"    #字符集
    chars = "abcdef"    #字符集
    vocab = {}                              #字典：字符与索引的对应关系
    for index, char in enumerate(chars):
        vocab[char] = index
    vocab['unk'] = len(char)
    return vocab


#判断是否是回文串
def isPalindrome(s: str) -> bool:
    n = len(s)
    dp = np.zeros((n,n))
    maxlength = 1

    for right in range(1, n):
        for left in range(0, right):
            if s[right] == s[left] and (right - left <=2 or dp[left+1][right-1]):
                dp[left][right] = 1
                if right-left+1 > maxlength:
                    maxlength = right-left+1
    if maxlength == n:
        return True
    else:
        return False


#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    x = [random.choice(x) for _ in range(sentence_length)]
    #指定哪些字出现时为正样本
    if isPalindrome(x):
        y = 1
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x,y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#建立模型
def build_model(sentence_length, vector_dim, vocab):
    model = TorchModel(sentence_length, vector_dim, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    sample_num = 500
    x, y = build_dataset(sample_num, vocab,sentence_length) #建立200个用于测试的样本
    print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred,y):
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
        print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
        return correct / (correct + wrong)

def main():
    epoch_num = 1000          #训练轮数
    batch_size = 200         #微批数量
    train_sample = 1000      #每轮训练总共训练的样本总数
    char_dim = 64           #字向量的维度
    sentence_length = 3     #样本文本长度
    learning_rate = 0.005   #学习率

    #建立字表
    vocab = build_vocab()
    #建立模型
    model = build_model(sentence_length, char_dim, vocab)
    #选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    #训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss= []
        for batch in range(train_sample // batch_size):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()   #梯度归零
            loss = model(x, y)  #计算损失
            loss.backward()     #计算梯度
            optim.step()        #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model,vocab,sentence_length)
        log.append([acc, np.mean(watch_loss)])

    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")     #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")    #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    #保存词表
    writer = open("vocab.json", "w", encoding="utf-8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 64  # 每个字的维度
    sentence_length = 3  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf-8")) #加载字典表
    model = build_model(sentence_length,char_dim,vocab)         #建立模型
    model.load_state_dict(torch.load(model_path))               #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])    #将输入序列化
    model.eval()    #验证模式
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i])) #打印结果

if __name__ == "__main__":
    main()
    input_strings = ["abc", "aaa","abb","cdc","ddd","fef"]
    predict(model_path='model.pth',vocab_path='vocab.json', input_strings=input_strings)







