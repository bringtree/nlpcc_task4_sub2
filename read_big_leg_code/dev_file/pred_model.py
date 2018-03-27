# coding=utf-8
# @author: cer
import tensorflow as tf
from utils import *
from model import Model
from my_metrics import *
from tensorflow.python import debug as tf_debug
import numpy as np
from utils import *
import fastText as fasttext
import os
import pickle

ckpt_path = './ckpt2/'
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
enable_w2v = True

train_X = np.load('train_input.npy')
train_slot_sentences = np.load('train_slot.npy')
train_Y = np.load('train_intent.npy')

test_X = np.load('test_input.npy')
test_slot_sentences = np.load('test_slot.npy')
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

with open("train_data_ed.pkl", "rb") as fp:
    train_data_ed = pickle.load(fp)

with open("test_data_ed.pkl", "rb") as fp:
    test_data_ed = pickle.load(fp)

sentence_result = [" ".join(v[0]) for v in test_data_ed]
result = [v[2] for v in test_data_ed]
# # # 校对文本
# with open("check", "w") as fp:
#     fp.writelines(result)

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

# [self.vocab_size, self.embedding_size]
if enable_w2v is True:
    with open("embedding_W.pkl", "rb") as fp:
        embedding_W = pickle.load(fp)
        embedding_W = tf.cast(embedding_W, tf.float32)
else:
    embedding_W = None


def get_model():
    model = Model(input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                  intent_size, epoch_num, batch_size, embedding_W)
    model.build()
    return model


def train(is_debug=False):
    model = get_model()
    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    saver.restore(sess, os.path.join(ckpt.model_checkpoint_path))
    length = int(len(index_test) / 64)
    pred_result = []
    for batch in (testBatch(batch_size, index_test, random_data=False)):
        decoder_prediction, intent = model.step(sess, batch, is_Test=True)
        decoder_prediction = np.transpose(decoder_prediction, [1, 0])
        for index in range(64):
            sen_len = batch[index][1]
            pred_result.append(index2intent[intent[index]])
            # print(index2intent[intent[index]])
    pred_result2 = [v for v in pred_result]
    with open("pred_result", "w") as fp:
        fp.writelines(pred_result2)
    different = []
    for i in range(len(pred_result2)):
        if result[i] != pred_result2[i]:
            print(sentence_result[i])
            print(result[i])
            print(pred_result2[i])


if __name__ == '__main__':
    train()
