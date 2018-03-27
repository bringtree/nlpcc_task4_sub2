# coding=utf-8
# @author: cer

import random
import numpy as np
import data_utils

flatten = lambda l: [item for sublist in l for item in sublist]  # 二维展成一维
index_seq2slot = lambda s, index2slot: [index2slot[i] for i in s]
index_seq2word = lambda s, index2word: [index2word[i] for i in s]


def data_pipeline(data, length=50, isTset=False):
    # data = [t[:-1] for t in data]  # 去掉'\n'
    # 数据的一行像这样：'BOS i want to fly from baltimore to dallas round trip EOS \tO O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight'
    # 分割成这样[原始句子的词，标注的序列，intent]

    seq_in, seq_out, intent = list(zip(*data))
    sin = []
    sout = []
    # padding，原始序列和标注序列结尾+<EOS>+n×<PAD>
    for i in range(len(seq_in)):
        temp = seq_in[i]
        if len(temp) < length:
            temp.append('<EOS>')
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sin.append(temp)

        temp = seq_out[i]
        if len(temp) < length:
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sout.append(temp)

        if isTset is False:
            data = list(zip(sin, sout, intent))
        else:
            data = list(zip(sin))
    return data


# 做一个字典
def get_info_from_training_data(data, test_data):
    seq_in, seq_out, intent = list(zip(*data))
    test_seq_in = list(zip(*test_data))
    vocab = set(flatten(seq_in))
    tmp_cab = set(flatten(test_seq_in[0]))
    vocab |= set(tmp_cab)
    slot_tag = set(flatten(seq_out))
    intent_tag = set(intent)
    # 生成word2index 句子字典
    word2index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    for token in vocab:
        if token not in word2index:
            word2index[token] = len(word2index)

    # 字典key value 翻转过来
    # 生成index2word
    index2word = {v: k for k, v in word2index.items()}

    # 生成tag2index 槽字典
    tag2index = {'<PAD>': 0, '<UNK>': 1, "O": 2}
    for tag in slot_tag:
        if tag not in tag2index:
            tag2index[tag] = len(tag2index)

    # 生成index2tag
    index2tag = {v: k for k, v in tag2index.items()}

    # 生成intent2index
    intent2index = {'<UNK>': 0}
    for ii in intent_tag:
        if ii not in intent2index:
            intent2index[ii] = len(intent2index)

    # 生成index2intent
    index2intent = {v: k for k, v in intent2index.items()}
    return word2index, index2word, tag2index, index2tag, intent2index, index2intent


def getBatch(batch_size, train_data, random_data=True):
    if random_data == True:
        random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

def testBatch(batch_size, train_data, random_data=False):
    sindex = 0
    eindex = batch_size
    batch = []
    while eindex < len(train_data):
        batch.append(train_data[sindex:eindex])
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp

    return batch


def to_index(train, word2index, slot2index, intent2index, isTest=False):
    new_train = []

    if isTest is False:
        for sin, sout, intent in train:
            sin_ix = list(map(lambda i: word2index[i] if i in word2index else word2index["<UNK>"],
                              sin))
            true_length = sin.index("<EOS>")
            sout_ix = list(map(lambda i: slot2index[i] if i in slot2index else slot2index["<UNK>"],
                               sout))
            intent_ix = intent2index[intent] if intent in intent2index else intent2index["<UNK>"]
            new_train.append([sin_ix, true_length, sout_ix, intent_ix])
    else:
        for sin in train:
            sin_ix = list(map(lambda i: word2index[i] if i in word2index else word2index["<UNK>"], sin[0]))
            true_length = sin[0].index("<EOS>")
            new_train.append([sin_ix, true_length])
    return new_train
