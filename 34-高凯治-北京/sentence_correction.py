from collections import defaultdict
from ngram_language_model import NgramLanguageModel
"""
文本纠错demo
加载同音字字典
加载语言模型
基本原理：
对于文本中每一个字，判断在其同音字中是否有其他字，在替换掉该字时，能使得语言模型计算的成句概率提高
"""


class Corrector:
    def __init__(self, language_model):
        #语言模型
        self.language_model = language_model
        #候选字字典
        self.sub_dict = self.load_tongyinzi("tongyin.txt")
        #成句概率的提升超过阈值则保留修改
        self.threshold = 7

    #实际上不光是同音字，同形字等也可以加入，本质上是常用的错字
    def load_tongyinzi(self, path):
        corpus = open(path, encoding="utf8").readlines()
        vocab = defaultdict(list)
        for line in corpus:
            new_line = line.strip()
            line_list = new_line.split(" ")
            for char in line_list[1]:
                vocab[line_list[0]].append(char)
        return vocab
    #纠错逻辑
    def correction(self, string):
        for char in string:
            sentence_prob = []
            sentence_prob.append(self.language_model.predict(string))
            if self.sub_dict[char] != []:
                candidate_char = self.sub_dict[char]
                for c in candidate_char:
                    new_string = string.replace(char, c, 1)
                    sentence_prob.append(self.language_model.predict(new_string))
                margin = max(sentence_prob) - sentence_prob[0]
                if margin > self.threshold:
                    pos = sentence_prob.index(max(sentence_prob)) - 1
                    string = string.replace(char, candidate_char[pos], 1)
        return string


corpus = open("财经.txt", encoding="utf8").readlines()         #打开语料，读取文本的所有行，返回列表
lm = NgramLanguageModel(corpus, 3)                            #初始化3-gram语言模型

cr = Corrector(lm)                                            #使用语言模型初始化文本纠错函数
string = "每国货币政册空间不大"
fix_string = cr.correction(string)
print("修改前：", string)
print("修改后：", fix_string)