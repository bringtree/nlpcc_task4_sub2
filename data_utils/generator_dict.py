# 生成 2个字典 一个做序列化 另外一个做反序列化
# 其实 我在想另外一个问题 这么提前做 字典会不会有点不妥。因为毕竟不是所有词都能在字典中找到 这样 到时候就会就会出现很多是unk 而其实在10g词向量里是有的。
# 或者 我们只做做 意图识别输出的字典？
# 字典是从1开始的
import os
# 使用方法
from pyltp import Segmentor
# import jieba
import numpy as np
import pickle

# 读取模型和切词的字典
# 存在问题 英文歌曲名等等 依旧 会被切开 比如 dave ramone 会被切成 dave ramone 2个单词
segmentor = Segmentor()
segmentor.load_with_lexicon("../ltp_data_v3.4.0/cws.model", "../data/words_list.txt")
# with open("../data/words_list.txt", "r") as fp:
#     for v in fp.readlines():
#         jieba.add_word(v.replace("\n", ''))

with open("../data/corpus.train.txt") as fp:
    context = fp.readlines()

paragraphs = []
paragraph = []
for sentence in context:
    if sentence is "\n":
        paragraphs.append(paragraph)
        paragraph = []
    else:
        paragraph.append(sentence)

for paragraphs_idx, paragraph in enumerate(paragraphs):
    for paragraph_idx, sentence in enumerate(paragraph):
        tmp = sentence.split("\t")[1:3]
        tmp[0] = segmentor.segment(tmp[0])
        tmp[0] = " ".join(tmp[0])
        # tmp[0] = "#".join(jieba.lcut(tmp[0]))

        # tmp[0] 代表的是句子
        # tmp[1] 代表意图(label)
        paragraphs[paragraphs_idx][paragraph_idx] = tmp

new_paragraph_sentences = []
new_paragraph_labels = []
for paragraph in paragraphs:
    new_sentences = []
    for sentence in paragraph:
        # new_sentences.append([word for word in sentence[0].split('#')])
        new_sentences.append([word for word in sentence[0].split(' ')])
    new_paragraph_sentences.append(new_sentences)

# 生成字典
all_words = []
for paragraph in new_paragraph_sentences:
    for sentence in paragraph:
        for word in sentence:
            all_words.append(word)

words = set(all_words)
words_len = len(words)
word_dict = {}
i = 1

for v in words:
    word_dict[v] = i
    i += 1

word_dict_reverse = dict(zip(word_dict.values(), word_dict.keys()))

with open("../data/word_dict.pkl", "wb") as fp:
    pickle.dump(word_dict, fp)

with open("../data/word_dict_reverse.pkl", "wb") as fp:
    pickle.dump(word_dict_reverse, fp)
