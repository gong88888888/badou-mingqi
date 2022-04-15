
from ngram_language_model import NgramLanguageModel


def get_tongyin(path):
    tongyin_dict = {}
    with open(path, encoding="utf8") as f:
        for index, sentence in enumerate(f.readlines()):
            splits = sentence.strip().split(" ")
            if len(splits) == 2:
                tongyin_dict[splits[0]] = splits[1]
    return tongyin_dict


def autofixsentence(sentence, tongyin_dict, lm):
    print("autofixsentence sentence: ", sentence)
    sentence_ppl = lm.calc_sentence_ppl(sentence)
    for index, char in enumerate(sentence):
        #print("char: ", char, " index: ", index)
        if char in tongyin_dict.keys():
            #print("tongyin: ", tongyin_dict[char])
            for tongyin_char in tongyin_dict[char]:
                new_sentence = sentence[0:index] + tongyin_char + sentence[index+1:]
                #print("new_sentence: ", new_sentence)
                new_sentence_ppl = lm.calc_sentence_ppl(new_sentence)
                #print("new_sentence_ppl: ", new_sentence_ppl)
                if sentence_ppl > new_sentence_ppl:
                    if sentence_ppl - new_sentence_ppl > 6:
                        print("find ", char, " rename to ", tongyin_char, " new sentence: ", new_sentence)
                        sentence = new_sentence
                        sentence_ppl = new_sentence_ppl
    return sentence



if __name__ == "__main__":
    tongyin_dict = get_tongyin("tongyin.txt")
    corpus = open("财经.txt", encoding="utf8").readlines()
    #corpus = open("test.txt", encoding="utf8").readlines()
    #print(corpus)
    #print(len(tongyin_dict))
    lm = NgramLanguageModel(corpus, 3)
    #print("词总数:", lm.ngram_count_dict[0])
    #print(len(lm.ngram_count_prob_dict))

    #sentence = "基晋公司进行企叶年金业务风圣水起"
    sentence = "每国货币政册空间不大"
    #print(lm.calc_sentence_ppl(sentence))
    sentence_ppl = lm.calc_sentence_ppl(sentence)
    print(autofixsentence(sentence, tongyin_dict, lm))
    #print(tongyin_dict)

