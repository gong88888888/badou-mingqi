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
    def __init__(self, language_model: NgramLanguageModel):
        # 语言模型
        self.language_model: NgramLanguageModel = language_model
        # 候选字字典
        self.sub_dict = self.load_tongyinzi("tongyin.txt")
        # 成句概率的提升超过阈值则保留修改
        self.threshold = 7

    # 实际上不光是同音字，同形字等也可以加入，本质上是常用的错字
    def load_tongyinzi(self, path):
        homophones = defaultdict(list)
        for line in open(path, encoding="utf8"):
            chines_chr, homos = line.split()
            homophones[chines_chr] = list(homos)
        return homophones

    # 纠错逻辑
    def correction(self, string):
        max_prob = self.language_model.predict(string)
        max_prob_sen = string

        for index, ch in enumerate(string):
            if ch in self.sub_dict:
                homos = self.sub_dict[ch]
                for homo in homos:
                    new_str = max_prob_sen[:index] + homo + max_prob_sen[index + 1 :]
                    prob = self.language_model.predict(new_str)
                    if prob - max_prob > self.threshold:
                        # print(f"{ch}->{homo},{max_prob}->{prob}")
                        max_prob = prob
                        max_prob_sen = new_str
        return max_prob_sen


corpus = open("财经.txt", encoding="utf8").readlines()
lm = NgramLanguageModel(corpus, 3)

cr = Corrector(lm)
string = "每国货币政册空间不大"
fix_string = cr.correction(string)
print("修改前：", string)
print("修改后：", fix_string)
