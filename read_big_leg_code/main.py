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
import pickle

# 这一块是配置的一些信息
# ckpt 这个是模型保存的地址
ckpt_path = './ckpt2/'
# 输入的序列长度 或者说时间步数（要改。改成自己识别长度）
input_steps = 30
# embedding的向量长度
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
# 是否用w2v 初始化embedding (有bug)
enable_w2v = True

# 项目的目录
os.chdir("/Users/huangpeisong/Desktop/task-slu-tencent.dingdang/rnn/read_big_leg_code")

# 训练集
train_X = np.load('train_input.npy')
train_slot_sentences = np.load('train_slot.npy')
train_Y = np.load('train_intent.npy')

# 测试集
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

# train_data_ed = data_pipeline(train_data, length=input_steps)
# test_data_ed = data_pipeline(test_data, length=input_steps)

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
index_test = to_index(test_data_ed, word2index, slot2index, intent2index)

# index_test = to_index(test_data_ed, word2index, slot2index, intent2index, isTest=True)


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
    # 先生成模型
    model = get_model()
    # session。。。
    sess = tf.Session()
    # 这个是要用做可视化的，导出到model_graph
    writer = tf.summary.FileWriter("./model_graph")
    # 把模型图写入到tensorboard
    writer.add_graph(sess.graph)
    # tf的调试器。可以 看参数的值
    if is_debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #     sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    # 初始化所有 参数。如果是导入模型的话这句语句 记得删除
    sess.run(tf.global_variables_initializer())
    # 收集所有操作op。 等等 到tensorboard。也是可视化那一块
    merged_summary = tf.summary.merge_all()
    # 模型保存最近3次
    saver = tf.train.Saver(max_to_keep=3)
    for epoch in range(epoch_num):
        train_loss = 0.0
        for i, batch in enumerate(testBatch(batch_size, index_train)):
            # 执行一个batch的训练
            # 进入step看具体。_这个是梯度下降。loss 是loss。
            # decoder_prediction 是预测的槽输出。intent是预测的意图。mask好像是pad。这些就是他不等于0 就是true 然后再 转成1 。
            _, loss, decoder_prediction, intent, mask = model.step(sess, batch)
            train_loss += loss
        print("[Epoch {}] Average train loss: {}".format(epoch, train_loss / i))
        # 每五次保存一次
        if (epoch % 5 == 0):
            saver.save(sess, ckpt_path, global_step=epoch)

        # 每训一个epoch，测试一次
        pred_slots = []
        slot_accs = []
        intent_accs = []
        for batch in testBatch(batch_size, index_test, random_data=False):
            decoder_prediction, intent = model.step(sess, batch, is_Test=True)

            decoder_prediction = np.transpose(decoder_prediction, [1, 0])
            for index in range(64):
                sen_len = batch[index][1]
                print("Input Sentence        : ", index_seq2word(batch[index][0], index2word)[:sen_len])
                print("Slot Truth            : ", index_seq2slot(batch[index][2], index2slot)[:sen_len])
                print("Slot Prediction       : ",
                      index_seq2slot(decoder_prediction[index], index2slot)[:sen_len])
                print("Intent Truth          : ", index2intent[batch[index][3]])
                print("Intent Prediction     : ", index2intent[intent[index]])
            slot_pred_length = list(np.shape(decoder_prediction))[1]
            pred_padded = np.lib.pad(decoder_prediction, ((0, 0), (0, input_steps - slot_pred_length)),
                                     mode="constant", constant_values=0)
            pred_slots.append(pred_padded)
        #     # print("slot_pred_length: ", slot_pred_length)


            # 这块计算准确的代码 我总觉得 有什么问题 但是 就是感觉不出来。。。
            true_slot = np.array((list(zip(*batch))[2]))
            true_length = np.array((list(zip(*batch))[1]))
            true_slot = true_slot[:, :slot_pred_length]
        #     # print(np.shape(true_slot), np.shape(decoder_prediction))
        #     # print(true_slot, decoder_prediction)
            slot_acc = accuracy_score(true_slot, decoder_prediction, true_length)
            intent_acc = accuracy_score(list(zip(*batch))[3], intent)
        #     # print("slot accuracy: {}, intent accuracy: {}".format(slot_acc, intent_acc))
            slot_accs.append(slot_acc)
            intent_accs.append(intent_acc)
        # pred_slots_a = np.vstack(pred_slots)
        # # print("pred_slots_a: ", pred_slots_a.shape)
        # true_slots_a = np.array(list(zip(*index_test))[2])[:pred_slots_a.shape[0]]
        # # print("true_slots_a: ", true_slots_a.shape)
        # # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


        # 这里有个操作 result_board 是要让他收集到 tensorboard。 然后我 在model中留了2个 scalar 就是会被收集进去。也就是result_result = [scalar1,scalar2]
        # 就是这两个scalar
        # tf.summary.scalar("intent_acc", self.intent_accs_op)
        # tf.summary.scalar("slot_acc", self.slot_accs_op)
        # 写入到writer里
        # writer.add_summary(result_board, epoch)
        intent_accuracy, slot_accuracy, result_board = model.get_score(sess, intent_accs, slot_accs, merged_summary)
        writer.add_summary(result_board, epoch)
        #
        # print("Intent accuracy for epoch {}: {}".format(epoch, intent_accuracy))
        # print("Slot accuracy for epoch {}: {}".format(epoch, slot_accuracy))
        # print("Slot F1 score for epoch {}: {}".format(epoch, f1_for_sequence_batch(true_slots_a, pred_slots_a)))



if __name__ == '__main__':
    train()
