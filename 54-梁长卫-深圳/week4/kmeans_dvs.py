import math
from collections import defaultdict

import jieba
import numpy as np
import random
import sys

class KMeansClusterer:  # k均值聚类
    def __init__(self, sentences, cluster_num):
        self.sentences = sentences
        self.cluster_num = cluster_num
        print("cluster_num: ", cluster_num)
        self.points = self.__pick_start_point(sentences, cluster_num)

    def cluster(self):
        results = []
        #print("self.points： ", self.points)
        for i in range(self.cluster_num):
            results.append([])
        for sentence in self.sentences:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(sentence, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            results[index] = results[index] + [sentence]
        #print("len(results): ", len(results))
        new_center = []
        for item in results:
            new_center.append(self.__center(item))
        #print("new_center: ", len(new_center), " new_center: ", new_center)
        if self.points == new_center:
            #sum = self.__sumdis(result)
            return results, self.points
        self.points = new_center
        return self.cluster()


    def __pick_start_point(self, sentences, cluster_num):
        if cluster_num < 0 or cluster_num > len(sentences):
            raise Exception("簇数设置有误")
        indexes = random.sample(np.arange(0, len(sentences), step=1).tolist(), cluster_num)
        #print(indexes)
        points = []
        for index in indexes:
            #print("senteces[", index, "]: ", sentences[index])
            points.append(sentences[index])
        #print("points: ", points)
        return points

    def __distance(self, str_a, str_b):
        str_a = str_a.lower()
        str_b = str_b.lower()
        #print("str_a: ", str_a)
        #print("str_b: ", str_b)
        matrix_ed = np.zeros((len(str_a)+1, len(str_b)+1), dtype=int)
        #print(matrix_ed)
        matrix_ed[0] = np.arange(len(str_b)+1)
        matrix_ed[:, 0] = np.arange(len(str_a)+1)
        #print(matrix_ed)
        for i in range(1, len(str_a)+1):
            for j in range(1, len(str_b)+1):
                dist_1 = matrix_ed[i-1][j] + 1
                dist_2 = matrix_ed[i][j-1] + 1
                dist_3 = matrix_ed[i-1][j-1] + (1 if str_a[i-1] != str_b[j-1] else 0)
                matrix_ed[i][j] = np.min([dist_1, dist_2, dist_3])
        #print(matrix_ed)
        return matrix_ed[-1][-1]

    def __center(self, list):
        #print("len: ", len(list))

        if len(list) <= 2:
            return list[0]

        distance_min = sys.maxsize
        new_center = -1
        #print("len(item)： ", len(list))
        for i, start in enumerate(list):
            sumis = 0
            for j, end in enumerate(list):
                if start != end:
                    sumis += self.__distance(start, end)
                    #print("i: ", i, " j: ", j, " sumis: ", sumis)
            #print("sumis: ", sumis, " distance_min: ", distance_min)
            if sumis < distance_min:
                new_center = i
                distance_min = sumis
        #print("find new_center: ", new_center, " sumis: ", sumis)
        return list[new_center]



def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            if len(sentences) < 200:
                sentences.add(sentence)
            #sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences


def main():
    sentences = load_sentence("titles.txt")  #加载所有标题
    print(len(sentences))
    sentences_array = []
    for index, sentence in enumerate(sentences):
        sentences_array.append(sentence)

    n_clusters = int(math.sqrt(len(sentences)))
    kmeans = KMeansClusterer(sentences_array, n_clusters)
    result, centers = kmeans.cluster()
    print("result: ", result)
    print("centers: ", centers)

    for index, list in enumerate(result):
        print("cluster %s :" % (index+1))
        for i in range(min(10, len(list))):  #随便打印几个，太多了看不过来
            print(list[i].replace(" ", ""))
        print("---------")

    #for index, sentence in enumerate(sentences):
        #print(sentence, " len(sentence): ", len(sentence))


    #print(dev("love", "lolpe"))

if __name__ == '__main__':
    main()
