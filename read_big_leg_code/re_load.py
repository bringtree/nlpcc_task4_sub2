# coding=utf-8
# @author: cer
import tensorflow as tf
from data import *
from model import Model
from my_metrics import *
from tensorflow.python import debug as tf_debug
import numpy as np
from data_utils import k_fold
import fastText as fasttext
import os

ckpt_path = '/Users/huangpeisong/Desktop/task-slu-tencent.dingdang/rnn/ckpt2'
input_steps = 30
# lstm输出的槽值大小
slot_size = input_steps
embedding_size = 300
# lstm 隐藏层单元参数的大小
hidden_size = 100
# 一个batch输入的64个句子
batch_size = 64
# 这个vocab_size 到时候 会直接就换成 那个很大很大的词向量来处理

# 迭代多少次
epoch_num = 20
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
test_data_ed = data_pipeline(test_data, length=input_steps)

# 生成6份字典 分别是 句子中的词 -> 序号  序号-> 句子中的词   (slot intent 同理)
# 这里加入测试集 只是为了 给测试集的句子中的词 也有一个编号而已
word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
    get_info_from_training_data(train_data_ed, test_data_ed)
intent_size = len(intent2index)
vocab_size = len(word2index)
# 接下来 完成 编码
index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
# index_test = to_index(test_data_ed, word2index, slot2index, intent2index, isTest=True)

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


def train():
    model = get_model()
    sess = tf.Session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(ckpt_path)

    # for epoch in range(epoch_num):
    #     mean_loss = 0.0
    #     train_loss = 0.0
    #     for i, batch in enumerate(getBatch(batch_size, index_train,random_data=False)):
    #         # 执行一个batch的训练
    #         _, loss, decoder_prediction, intent, mask = model.step(sess, batch)
    #         mean_loss += loss
    #         train_loss += loss
    #         if i % 10 == 0:
    #             if i > 0:
    #                 mean_loss = mean_loss / 10.0
    #             print('~~~~~~~~Average train loss at epoch %d, step %d: %f' % (epoch, i, mean_loss))
    #             mean_loss = 0
    #     train_loss /= (i + 1)
    #     # print("[Epoch {}] Average train loss: {}".format(epoch, train_loss))
    #     saver.save(sess, ckpt_path+"/model", global_step=epoch)
    #
    # ckpt = tf.train.get_checkpoint_state(ckpt_path)

    saver.restore(sess, ckpt.model_checkpoint_path)

    for epoch in range(epoch_num):
        mean_loss = 0.0
        train_loss = 0.0
        for i, batch in enumerate(getBatch(batch_size, index_train, random_data=False)):
            # 执行一个batch的训练
            _, loss, decoder_prediction, intent, mask = model.step(sess, batch)
            mean_loss += loss
            train_loss += loss
            if i % 10 == 0:
                if i > 0:
                    mean_loss = mean_loss / 10.0
                print('~~~~~~~~Average train loss at epoch %d, step %d: %f' % (epoch, i, mean_loss))
                mean_loss = 0
        train_loss /= (i + 1)
        # print("[Epoch {}] Average train loss: {}".format(epoch, train_loss))
        # saver.save(sess, ckpt_path+"/model", global_step=epoch)


if __name__ == '__main__':
    train()
