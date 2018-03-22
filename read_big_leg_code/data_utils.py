import numpy as np
import jieba
from sklearn.model_selection import StratifiedKFold
import random


# K折
def k_fold(k, X, Y, slot_sentences=None):
    """
    按照数据的标签比值分割数据
    用于生成训练出训练集合测试集数据
    :param k:int K折,
    :param X:list 特征,
    :param Y:list 标签
    :return:训练集的特征，训练集的标签，和测试集的特征，测试集的标签
    """

    sfolder = StratifiedKFold(n_splits=k, random_state=random.seed(), shuffle=True)
    train_index = []
    test_index = []

    train_X = []
    train_slot_sentences = []
    train_Y = []
    test_X = []
    test_slot_sentences = []
    test_Y = []

    for x, y in sfolder.split(X, Y):
        train_index.append(x)
        test_index.append(y)

    for i in range(len(test_index)):
        test_X.append([X[v] for v in test_index[i]])
        test_Y.append([Y[v] for v in test_index[i]])
        if slot_sentences:
            test_slot_sentences.append([slot_sentences[v] for v in test_index[i]])
    for i in range(len(train_index)):
        train_X.append([X[v] for v in train_index[i]])
        train_Y.append([Y[v] for v in train_index[i]])
        if slot_sentences:
            train_slot_sentences.append([slot_sentences[v] for v in train_index[i]])
    if slot_sentences:
        return train_X, train_slot_sentences, train_Y, test_X, test_slot_sentences, test_Y
    return train_X, train_Y, test_X, test_Y


with open("test_data.txt") as fp:
    raw_data = [v.split(' ') for v in fp.readlines()]

sentences = [v[1:v.index("EOS")] for v in raw_data]
slot_sentences = [v[v.index("EOS") + 1:-1] for v in raw_data]
labels = [v[-1].replace('\n', '') for v in raw_data]

train_test_X, train_test_slot_sentences, train_test_Y, dev_X, dev_slot_sentences, dev_Y = k_fold(5, X=sentences,
                                                                                                 Y=labels,
                                                                                                 slot_sentences=slot_sentences)
train_X, train_slot_sentences, train_Y, test_X, test_slot_sentences, test_Y = k_fold(10, X=train_test_X[0],
                                                                                     Y=train_test_Y[0],
                                                                                     slot_sentences=slot_sentences)

