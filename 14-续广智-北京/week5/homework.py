from ngram_language_model import NgramLanguageModel

class Corrector(object):
    def __init__(self, tongyin_file, lm, ppl_rtol=0.2):
        """同音字更正器

        Args:
            tongyin_file (str): path to tongyin char table .
            lm (NgramLanguageModel): language model.
            pl_rtol (float): minimum relative sentence perplexity drop required
                to correct a char.
        """
        self.tongyin_dict = self.load_tongyin_file(tongyin_file)
        self.lm = lm
        self.ppl_rtol = ppl_rtol

    def load_tongyin_file(self, file_name):
        '''Load tongyin char table'''
        res = {}
        with open(file_name, 'r', encoding='utf-8') as fin:
            for line in fin.readlines():
                char, replaces = line.strip().split(' ')
                res[char] = list(replaces)
        return res

    def correct_sentence(self, sentence):
        '''Correct tongyin chars in a sentence'''

        sen_list = list(sentence)
        old_ppl = lm.calc_sentence_ppl(sentence)
        print('Old sentence perplexity:', old_ppl)

        for ii, charii in enumerate(sen_list):
            repsii = self.tongyin_dict.get(charii)
            if repsii is not None:
                pplii = []
                for repjj in repsii:
                    sen_list[ii] = repjj
                    ppljj = self.lm.calc_sentence_ppl(''.join(sen_list))
                    pplii.append(ppljj)

                # find the lowest ppl after corrections
                min_ppl = min(pplii)
                if (old_ppl - min_ppl) / old_ppl >= self.ppl_rtol:
                    idx = pplii.index(min_ppl)
                    new_char = repsii[idx]
                    print('Correct:', charii, '->', new_char)
                    sen_list[ii] = new_char
                    # recompute ppl
                    old_ppl = lm.calc_sentence_ppl(''.join(sen_list))
                else:
                    sen_list[ii] = charii

        print('New sentence perplexity:', old_ppl)
        return ''.join(sen_list)

if __name__ == "__main__":

    corpus = open("corpus.txt", encoding="utf8").readlines()
    lm = NgramLanguageModel(corpus, 3)
    print("词总数:", lm.ngram_count_dict[0])
    corrector = Corrector('./tongyin.txt', lm, 0.2)

    sentence = "昨日伤海再度绷溃"
    new_sentence = corrector.correct_sentence(sentence)
    print('Old sentence:', sentence)
    print('New sentence:', new_sentence)
