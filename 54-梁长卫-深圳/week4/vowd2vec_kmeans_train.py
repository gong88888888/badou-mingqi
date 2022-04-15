#!/usr/bin/env python3
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences


def sentences_to_vectors(sentences, model):
    vectors = []
    print("sentences_to_vectors model.vector_size： ", model.vector_size)
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def distance(p1, p2):
    #计算两点间距
    tmp = 0
    for i in range(len(p1)):
        tmp += pow(p1[i] - p2[i], 2)
    return pow(tmp, 0.5)

def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化
    print(vectors, vectors.shape)

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("n_clusters: ", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    vector_label_dict = defaultdict(list)
    for vector, label in zip(vectors, kmeans.labels_):
        vector_label_dict[label].append(vector)

    print("len(vector_label_dict): ", len(vector_label_dict))
    sumdis_label_dict = defaultdict(list)
    for index, center in enumerate(kmeans.cluster_centers_):
        print("index: ", index, " center: ", center)
        sumdis = 0
        for point in vector_label_dict[index]:
            sumdis += distance(center, point)
        sumdis_label_dict[index] = sumdis/len(vector_label_dict[index])
        print("sumdis_label_dict[", index, "] total: ", sumdis, " count: ", len(vector_label_dict[index]), " avg: ", sumdis_label_dict[index])

    print("After sorted ")
    for index, sumdis in sorted(sumdis_label_dict.items(), key=lambda x: x[1]):
        print("sumdis_label_dict[", index, "]: ", sumdis)
        sentences = sentence_label_dict.get(index)
        print("cluster %s :" % index, " len: ", len(sentences))
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
