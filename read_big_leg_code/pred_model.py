# coding=utf-8
# @author: cer
import tensorflow as tf
from read_big_leg_code.data import *
from read_big_leg_code.model import Model
from read_big_leg_code.my_metrics import *
from tensorflow.python import debug as tf_debug
import numpy as np
from data_utils import k_fold
import fastText as fasttext
import os

ckpt_path = '/Users/huangpeisong/Desktop/task-slu-tencent.dingdang/rnn/ckpt'
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
epoch_num = 100
enable_w2v = False

train_X = np.load('/Users/huangpeisong/Desktop/task-slu-tencent.dingdang/rnn/read_big_leg_code/train_input.npy')
train_slot_sentences = np.load(
    '/Users/huangpeisong/Desktop/task-slu-tencent.dingdang/rnn/read_big_leg_code/train_slot.npy')
train_Y = np.load('/Users/huangpeisong/Desktop/task-slu-tencent.dingdang/rnn/read_big_leg_code/train_intent.npy')

test_X = np.load('/Users/huangpeisong/Desktop/task-slu-tencent.dingdang/rnn/read_big_leg_code/test_input.npy')
test_slot_sentences = np.load(
    '/Users/huangpeisong/Desktop/task-slu-tencent.dingdang/rnn/read_big_leg_code/test_slot.npy')
test_Y = np.load('/Users/huangpeisong/Desktop/task-slu-tencent.dingdang/rnn/read_big_leg_code/test_intent.npy')

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
test_data_ed = data_pipeline(test_data, length=input_steps, isTset=True)
word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
    get_info_from_training_data(train_data_ed, test_data_ed)
intent_size = len(intent2index)
vocab_size = len(word2index)
# 接下来 完成 编码
index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
index_test = to_index(test_data_ed, word2index, slot2index, intent2index, isTest=True)

# [self.vocab_size, self.embedding_size]
if enable_w2v is True:
    w2v_model_bin = "/home/bringtree/data/wiki.zh.bin"
    w2v_model = fasttext.load_model(w2v_model_bin)
    embedding_W = np.eye(vocab_size, embedding_size)
    for key, value in word2index.items():
        embedding_W[value] = w2v_model.get_word_vector(key)
else:
    embedding_W = None


def get_model():
    model = Model(input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                  intent_size, epoch_num, batch_size, embedding_W)
    model.build()
    return model


def print_predict():
    model = get_model()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_path)

    saver.restore(sess, os.path.join(ckpt.model_checkpoint_path))
    for j, batch in enumerate(getBatch(batch_size, index_train, random_data=False)):
        train_op, loss, decoder_prediction, intent, mask = model.step(sess, batch)
        decoder_prediction = np.transpose(decoder_prediction, [1, 0])
        if j == 0:
            # index = random.choice(range(len(batch)))
            for index in range(64):
                sen_len = batch[index][1]
                print("Input Sentence        : ", index_seq2word(batch[index][0], index2word)[:sen_len])
                print("Slot Prediction       : ", index_seq2slot(decoder_prediction[index], index2slot)[:sen_len])
                print("Intent Prediction     : ", index2intent[intent[index]])


if __name__ == '__main__':
    print_predict()
