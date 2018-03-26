from data import *
import numpy as np
import os
from data_utils import k_fold

input_steps = 30
with open("/Users/huangpeisong/Desktop/task-slu-tencent.dingdang/rnn/test_data.txt") as fp:
    raw_data = [v.split(' ') for v in fp.readlines()]

sentences = [v[1:v.index("EOS")] for v in raw_data]
slot_sentences = [v[v.index("EOS") + 2:-1] for v in raw_data]
labels = [v[-1].replace('\n', '') for v in raw_data]
train_X, train_slot_sentences, train_Y, test_X, test_slot_sentences, test_Y = k_fold(10, X=sentences, Y=labels,
                                                                                     slot_sentences=slot_sentences)
train_X = train_X[0]
train_slot_sentences = train_slot_sentences[0]
train_Y = train_Y[0]

test_X = test_X[0]
test_slot_sentences = test_slot_sentences[0]
test_Y = test_Y[0]

train_data = []
for idx in range(len(train_X)):
    tmp = []
    tmp.append(train_X[idx])
    tmp.append(train_slot_sentences[idx])
    tmp.append(train_Y[idx])
    train_data.append(tmp)

test_data = []
for idx in range(len(test_X)):
    tmp = []
    tmp.append(test_X[idx])
    tmp.append(test_slot_sentences[idx])
    tmp.append(test_Y[idx])
    test_data.append(tmp)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

train_data_ed = data_pipeline(train_data, length=input_steps)
test_data_ed = data_pipeline(test_data, length=input_steps)

# 生成6份字典 分别是 句子中的词 -> 序号  序号-> 句子中的词   (slot intent 同理)
# 这里加入测试集 只是为了 给测试集的句子中的词 也有一个编号而已
word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
    get_info_from_training_data(train_data_ed, test_data_ed)

# 保存字典
np.save("index2word_dict.npy", index2word)
np.save("index2slot_dict.npy", index2slot)
np.save("index2intent_dict.npy", index2intent)

np.save("word2index_dict.npy", word2index)
np.save("slot2index_dict.npy", slot2index)
np.save("intent2index_dict.npy", intent2index)


# 测试
# word2index = np.load("word2index_dict.npy")
# index2word = np.load("index2word_dict.npy")
# slot2index = np.load("slot2index_dict.npy")
# index2slot = np.load("index2slot_dict.npy")
# intent2index = np.load("intent2index_dict.npy")
# index2intent = np.load("index2intent_dict.npy")
# index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
# index_test = to_index(test_data_ed, word2index, slot2index, intent2index, isTest=True)
