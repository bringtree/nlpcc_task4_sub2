# 给所有输入编码
import os
# 使用方法
# from pyltp import Segmentor
import numpy as np
import pickle
import keras
import jieba
######################################################################
with open("../data/word_dict.pkl", "rb") as fp:
    word_dict = pickle.load(fp)

with open("../data/word_dict_reverse.pkl", "rb") as fp:
    word_dict_reverse = pickle.load(fp)

with open("../data/labels.txt") as fp:
    labels_type = [v[:-1] for v in fp.readlines()]

label_dict = {}
i = 1

for v in labels_type:
    label_dict[v] = i
    i += 1

label_dict_reverse = dict(zip(label_dict.values(), label_dict.keys()))

######################################################################

# 读取模型和切词的字典
# segmentor = Segmentor()
# segmentor.load_with_lexicon("../ltp_data_v3.4.0/cws.model", "../data/words_list.txt")
with open("../data/words_list.txt", "r") as fp:
    for v in fp.readlines():
        jieba.add_word(v.replace("\n", ''))

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

label_list = {
    "OTHERS": 0,
    "music.next": 0,
    "music.pause": 0,
    "music.play": 0,
    "music.prev": 0,
    "navigation.cancel_navigation": 0,
    "navigation.navigation": 0,
    "navigation.open": 0,
    "navigation.start_navigation": 0,
    "phone_call.cancel": 0,
    "phone_call.make_a_phone_call": 0
}

for paragraphs_idx, paragraph in enumerate(paragraphs):
    for paragraph_idx, sentence in enumerate(paragraph):
        tmp = sentence.split("\t")[1:3]
        # tmp[0] = segmentor.segment(tmp[0])
        # tmp[0] = " ".join(tmp[0])
        tmp[0] = "#".join(jieba.lcut(tmp[0]))
        # tmp[0] 代表的是句子
        # tmp[1] 代表意图(label)
        label_list[str(tmp[1])] +=1
        paragraphs[paragraphs_idx][paragraph_idx] = tmp

new_paragraph_sentences = []
new_paragraph_labels = []
for paragraph in paragraphs:
    new_sentences = []
    new_labels = []
    new_paragraph = []
    for sentence in paragraph:
        # 对输入的句子编码
        new_sentences.append([word_dict[word] for word in sentence[0].split('#')])
        # 对输入的句子不编码
        # new_sentences.append([word for word in sentence[0].split(' ')])

        new_labels.append(label_dict[sentence[1]])

    new_paragraph_sentences.append(new_sentences)
    new_paragraph_labels.append(new_labels)

#####################################################################
# 填充0 每个上下文长度控制在30 每个句子长度控制在30
pad_new_paragraph_labels = keras.preprocessing.sequence.pad_sequences(new_paragraph_labels, maxlen=30, dtype='int32',
                                                                      padding='post', truncating='post', value=0.)

for idx, new_paragraph in enumerate(new_paragraph_sentences):
    new_paragraph_sentences[idx] = keras.preprocessing.sequence.pad_sequences(new_paragraph, maxlen=30, dtype='int32',
                                                                              padding='post', truncating='post',
                                                                              value=0.)

for idx, new_paragraph in enumerate(new_paragraph_sentences):
    if (len(new_paragraph_sentences[idx]) > 30):
        print('长度不够用')
    for i in range(len(new_paragraph_sentences[idx]), 30):
        new_paragraph_sentences[idx] = np.row_stack((new_paragraph_sentences[idx], np.zeros(30, dtype="int8")))

np.save("../data/pad_new_paragraph_labels.npy", pad_new_paragraph_labels)
np.save("../data/new_paragraph_sentences.npy", new_paragraph_sentences)
######################################################################

####################################################################
# 生成字典
# all_words = []
# for paragraph in new_paragraph_sentences:
#     for sentence in paragraph:
#         tmp = sentence.split(' ')
#         for word in tmp:
#             all_words.append(word)
#
# words = set(all_words)
# words_len = len(words)
# word_dict = {}
# i = 1
#
# for v in words:
#     word_dict[v] = i
#     i += 1
#
# word_dict_reverse = dict(zip(word_dict.values(), word_dict.keys()))
#
# with open("word_dict.pkl", "wb") as fp:
#     pickle.dump(word_dict, fp)
#
# with open("word_dict_reverse.pkl", "wb") as fp:
#     pickle.dump(word_dict_reverse, fp)
######################################################################

######################################################################
# 切batch_size
# batch_size = 20
# X_batches = []
#
# Y_batches = []
#
# begin_index = 0
# end_index = batch_size
# while end_index < len(new_paragraph_sentences):
#     X_batches.append(new_paragraph_sentences[begin_index:end_index])
#     Y_batches.append(pad_new_paragraph_labels[begin_index:end_index])
#     begin_index = end_index
#     end_index = end_index + batch_size
#
# # 统计个数 有问题！。 现在单个 到时候 传入数据在搞就ok。所以 把代码写在了训练的文件中
# X_batches_0 = X_batches[0]


# 7,2
# word_inputs_actual_length = np.array([
#         [8, 8, 8, 8, 8, 8, 8],
#         [5, 6, 7, 8, 0, 0, 0]
#     ])

# def non_zero_times_count(sentence):
#     num = 0
#     for v in sentence:
#         if v != 0:
#             num += 1
#     return num
#
#
# word_inputs_actual_length = []
# for paragraph_idx, paragraph in enumerate(X_batches_0):
#     # 每个wordinputlength 中要放入 15个句子的单词长度
#     tmp = np.zeros(30, dtype="int8")
#     for sentence_idx, sentence in enumerate(paragraph):
#         tmp[sentence_idx] = non_zero_times_count(sentence)
#
#     word_inputs_actual_length.append(tmp)
# word_inputs_actual_length = np.array(word_inputs_actual_length)
# # word_inputs_actual_length = np.transpose(word_inputs_actual_length, [1, 0])
#
#
# sentences_inputs_actual_num = [non_zero_times_count(v) for v in word_inputs_actual_length]
