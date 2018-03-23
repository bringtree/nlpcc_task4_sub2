# coding=utf-8
# @author: cer
import tensorflow as tf
from read_big_leg_code.data import *
from read_big_leg_code.model import Model
from read_big_leg_code.my_metrics import *
from tensorflow.python import debug as tf_debug
import numpy as np
from data_utils import k_fold

input_steps = 30
embedding_size = 100
hidden_size = 100
batch_size = 64
# 这个vocab_size 到时候 会直接就换成 那个很大很大的词向量来处理
vocab_size = 12021
slot_size = 30
epoch_num = 50

# 这块到时候替换掉 ~~~~~~~~~~~~~~~~
with open("/Users/huangpeisong/Desktop/task-slu-tencent.dingdang/rnn/read_big_leg_code/test_data.txt") as fp:
    raw_data = [v.split(' ') for v in fp.readlines()]

sentences = [v[1:v.index("EOS")] for v in raw_data]
slot_sentences = [v[v.index("EOS") + 2:-1] for v in raw_data]
labels = [v[-1].replace('\n', '') for v in raw_data]
train_X, train_slot_sentences, train_Y, test_X, test_slot_sentences, test_Y = k_fold(10, X=sentences, Y=labels,
                                                                                     slot_sentences=slot_sentences)

train_data = []
for idx in range(len(train_X[0])):
    tmp = []
    tmp.append(train_X[0][idx])
    tmp.append(train_slot_sentences[0][idx])
    tmp.append(train_Y[0][idx])
    train_data.append(tmp)

test_data = []
for idx in range(len(test_X[0])):
    tmp = []
    tmp.append(test_X[0][idx])
    tmp.append(test_slot_sentences[0][idx])
    tmp.append(test_Y[0][idx])
    test_data.append(tmp)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

train_data_ed = data_pipeline(train_data, length=input_steps)
test_data_ed = data_pipeline(test_data, length=input_steps)

# 生成6份字典 分别是 句子中的词 -> 序号  序号-> 句子中的词   (slot intent 同理)
word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
    get_info_from_training_data(train_data_ed)
intent_size = len(intent2index)

# 接下来 完成 编码
index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
index_test = to_index(test_data_ed, word2index, slot2index, intent2index)

def get_model():
    model = Model(input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                  intent_size, epoch_num, batch_size)
    model.build()
    return model


def train(is_debug=False):
    model = get_model()
    sess = tf.Session()
    if is_debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    sess.run(tf.global_variables_initializer())

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
        print("[Epoch {}] Average train loss: {}".format(epoch, train_loss))

        # 每训一个epoch，测试一次
        pred_slots = []
        slot_accs = []
        intent_accs = []
        for j, batch in enumerate(getBatch(batch_size, index_test)):

            _, loss, decoder_prediction, intent, mask = model.step(sess, batch)

            decoder_prediction = np.transpose(decoder_prediction, [1, 0])
            if j == 0:
                index = random.choice(range(len(batch)))
                # index = 0
                sen_len = batch[index][1]
                print("Input Sentence        : ", index_seq2word(batch[index][0], index2word)[:sen_len])
                print("Slot Truth            : ", index_seq2slot(batch[index][2], index2slot)[:sen_len])
                print("Slot Prediction       : ", index_seq2slot(decoder_prediction[index], index2slot)[:sen_len])
                print("Intent Truth          : ", index2intent[batch[index][3]])
                print("Intent Prediction     : ", index2intent[intent[index]])
            slot_pred_length = list(np.shape(decoder_prediction))[1]
            pred_padded = np.lib.pad(decoder_prediction, ((0, 0), (0, input_steps - slot_pred_length)),
                                     mode="constant", constant_values=0)
            pred_slots.append(pred_padded)
            # print("slot_pred_length: ", slot_pred_length)
            true_slot = np.array((list(zip(*batch))[2]))
            true_length = np.array((list(zip(*batch))[1]))
            true_slot = true_slot[:, :slot_pred_length]
            # print(np.shape(true_slot), np.shape(decoder_prediction))
            # print(true_slot, decoder_prediction)
            slot_acc = accuracy_score(true_slot, decoder_prediction, true_length)
            intent_acc = accuracy_score(list(zip(*batch))[3], intent)
            # print("slot accuracy: {}, intent accuracy: {}".format(slot_acc, intent_acc))
            slot_accs.append(slot_acc)
            intent_accs.append(intent_acc)
        pred_slots_a = np.vstack(pred_slots)
        # print("pred_slots_a: ", pred_slots_a.shape)
        true_slots_a = np.array(list(zip(*index_test))[2])[:pred_slots_a.shape[0]]
        # print("true_slots_a: ", true_slots_a.shape)
        print("Intent accuracy for epoch {}: {}".format(epoch, np.average(intent_accs)))
        print("Slot accuracy for epoch {}: {}".format(epoch, np.average(slot_accs)))
        print("Slot F1 score for epoch {}: {}".format(epoch, f1_for_sequence_batch(true_slots_a, pred_slots_a)))



if __name__ == '__main__':
    train()
