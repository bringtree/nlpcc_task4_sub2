# 用来生成 embedding中的词向量
from data import *
import fastText as fasttext
import os
import pickle

input_steps = 30
embedding_size = 300
# lstm 隐藏层单元参数的大小
hidden_size = 100
# 一个batch输入的64个句子
batch_size = 64
# 这个vocab_size 到时候 会直接就换成 那个很大很大的词向量来处理
# lstm输出的槽值大小
slot_size = 30
# 迭代多少次
epoch_num = 20
enable_w2v = False

train_X = np.load('train_input.npy')
train_slot_sentences = np.load(
    'train_slot.npy')
train_Y = np.load('train_intent.npy')

test_X = np.load('test_input.npy')
test_slot_sentences = np.load(
    'test_slot.npy')
test_Y = np.load('test_intent.npy')

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

with open("index2word_dict.pkl", "rb") as fp:
    index2word = pickle.load(fp)
with open("index2slot_dict.pkl", "rb") as fp:
    index2slot = pickle.load(fp)
with open("index2intent_dict.pkl", "rb") as fp:
    index2intent = pickle.load(fp)

with open("word2index_dict.pkl", "rb") as fp:
    word2index = pickle.load(fp)
with open("slot2index_dict.pkl", "rb") as fp:
    slot2index = pickle.load(fp)
with open("intent2index_dict.pkl", "rb") as fp:
    intent2index = pickle.load(fp)


intent_size = len(intent2index)
vocab_size = len(word2index)
# 接下来 完成 编码
index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
index_test = to_index(test_data_ed, word2index, slot2index, intent2index, isTest=True)

w2v_model_bin = "/home/bringtree/data/wiki.zh.bin"
w2v_model = fasttext.load_model(w2v_model_bin)
embedding_W = np.eye(vocab_size, embedding_size)
for key, value in word2index.items():
    embedding_W[value] = w2v_model.get_word_vector(key)

with open("embedding_W.pkl", "wb") as fp:
    pickle.dump(embedding_W, fp)
