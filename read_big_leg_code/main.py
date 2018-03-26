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
word2index = np.load("word2index_dict.npy")
index2word = np.load("index2word_dict.npy")
slot2index = np.load("slot2index_dict.npy")
index2slot = np.load("index2slot_dict.npy")
intent2index = np.load("intent2index_dict.npy")
index2intent = np.load("index2intent_dict.npy")

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


def train(is_debug=False):
    model = get_model()
    sess = tf.Session()
    writer = tf.summary.FileWriter("./model")
    writer.add_graph(sess.graph)
    # if is_debug:
    #     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #     sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    sess.run(tf.global_variables_initializer())
    merged_summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    for epoch in range(epoch_num):
        mean_loss = 0.0
        train_loss = 0.0
        for i, batch in enumerate(getBatch(batch_size, index_train)):
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

        saver.save(sess, ckpt_path, global_step=epoch)

        # 每训一个epoch，测试一次
        pred_slots = []
        slot_accs = []
        intent_accs = []
        if epoch == 19:
            for j, batch in enumerate(getBatch(batch_size, index_test, random_data=False)):

                decoder_prediction, intent = model.step(sess, batch, is_Test=True)
                # writer.add_summary(result_board, epoch)

                decoder_prediction = np.transpose(decoder_prediction, [1, 0])
                if j == 0:
                    # index = random.choice(range(len(batch)))
                    for index in range(64):
                        sen_len = batch[index][1]
                        print("Input Sentence        : ", index_seq2word(batch[index][0], index2word)[:sen_len])
                        # print("Slot Truth            : ", index_seq2slot(batch[index][2], index2slot)[:sen_len])
                        print("Slot Prediction       : ",
                              index_seq2slot(decoder_prediction[index], index2slot)[:sen_len])
                        # print("Intent Truth          : ", index2intent[batch[index][3]])
                        print("Intent Prediction     : ", index2intent[intent[index]])
        #     slot_pred_length = list(np.shape(decoder_prediction))[1]
        #     pred_padded = np.lib.pad(decoder_prediction, ((0, 0), (0, input_steps - slot_pred_length)),
        #                              mode="constant", constant_values=0)
        #     pred_slots.append(pred_padded)
        #     # print("slot_pred_length: ", slot_pred_length)
        #     true_slot = np.array((list(zip(*batch))[2]))
        #     true_length = np.array((list(zip(*batch))[1]))
        #     true_slot = true_slot[:, :slot_pred_length]
        #     # print(np.shape(true_slot), np.shape(decoder_prediction))
        #     # print(true_slot, decoder_prediction)
        #     slot_acc = accuracy_score(true_slot, decoder_prediction, true_length)
        #     intent_acc = accuracy_score(list(zip(*batch))[3], intent)
        #     # print("slot accuracy: {}, intent accuracy: {}".format(slot_acc, intent_acc))
        #     slot_accs.append(slot_acc)
        #     intent_accs.append(intent_acc)
        # pred_slots_a = np.vstack(pred_slots)
        # # print("pred_slots_a: ", pred_slots_a.shape)
        # true_slots_a = np.array(list(zip(*index_test))[2])[:pred_slots_a.shape[0]]
        # # print("true_slots_a: ", true_slots_a.shape)
        # # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #
        # intent_accuracy, slot_accuracy, result_board = model.get_score(sess, intent_accs, slot_accs, merged_summary)
        # writer.add_summary(result_board, epoch)
        #
        # print("Intent accuracy for epoch {}: {}".format(epoch, intent_accuracy))
        # print("Slot accuracy for epoch {}: {}".format(epoch, slot_accuracy))
        # print("Slot F1 score for epoch {}: {}".format(epoch, f1_for_sequence_batch(true_slots_a, pred_slots_a)))
        #


if __name__ == '__main__':
    train()
