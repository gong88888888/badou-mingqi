# -*- coding: utf-8 -*-

import jieba
from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
"""
@Project ：badou-mingqi 
@File    ：word2vector_kmeans.py
@Author  ：luigi
@Date    ：2022/4/11 5:14 PM 
"""

#输入模型文件路径
#加载训练好的模型
def load_word2Vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
        print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split() #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector +=model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector +=np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def main():
    model = load_word2Vec_model("model.w2v")
    sentences = load_sentence("title_20220408.txt")
    vectors = sentences_to_vectors(sentences, model)

    n_cluster = 25
    kmeans = KMeans(n_cluster)
    kmeans.fit(vectors)

    vector_label_dict = defaultdict(list)
    distances_label_dict = {}

    for vector, label in zip(vectors, kmeans.labels_):  #取出句子向量和标签
        vector_label_dict[label].append(np.linalg.norm(vector-kmeans.cluster_centers_[label])) #同标签的计算每个句子向量到质心的距离

    for label, distances in vector_label_dict.items():
        distances_label_dict[label] = sum(distances)/len(distances) #由于每个标签包含的句子数量不同，因此需要对距离求和平均

    distances_sorted_index = sorted(distances_label_dict, key=distances_label_dict.get) #按照distance的大小对dict进行排序
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for label in distances_sorted_index:
        print("cluster %s :" % label)
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):        #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == '__main__':
    main()




