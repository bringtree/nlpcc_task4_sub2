import numpy as np
import jieba
from sklearn.model_selection import StratifiedKFold
import random


# K折
def k_fold(k, X, Y):
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
    train_Y = []
    test_X = []
    test_Y = []

    for x, y in sfolder.split(X, Y):
        train_index.append(x)
        test_index.append(y)

    for i in range(len(test_index)):
        test_X.append([X[v] for v in test_index[i]])
        test_Y.append([Y[v] for v in test_index[i]])

    for i in range(len(train_index)):
        train_X.append([X[v] for v in train_index[i]])
        train_Y.append([Y[v] for v in train_index[i]])

    return train_X, train_Y, test_X, test_Y


if __name__ == '__main__':

    # 读取出没有处理过的句子和标签
    row_data = []
    with open("corpus.train.txt") as fp:
        tmp = fp.readlines()
        for v in tmp:
            if v == '\n':
                continue
            row_data.append(v.split('\t'))

    not_cut_sentences = [v[1] for v in row_data]
    label = [v[2] for v in row_data]

    # 添加专有名字到jieba中，并且分词
    slot_file = ['toplist.txt',
                 'theme.txt',
                 'style.txt',
                 'song.txt',
                 'singer.txt',
                 'scene.txt',
                 'language.txt',
                 'instrument.txt',
                 'emotion.txt',
                 'custom_destination.txt',
                 'age.txt']
    slot_dict = {}
    for v in slot_file:
        with open("slot/" + str(v)) as fp:
            tmp = [v.replace('\n', '') for v in fp.readlines()]
            type_set = set(tmp)
            slot_dict[v[:-4]] = type_set

    jieba.del_word('我们不一样')
    for v in slot_dict:
        for v_v in slot_dict[v]:
            jieba.add_word(v_v)
    print(jieba.lcut('我们不一样'))

    # end_senteces是全部的句子
    # label 是全部的意图
    cut_sentences = []
    for v in not_cut_sentences:
        cut_sentences.append(jieba.lcut(v))
    train_test_X, train_test_Y, dev_X, dev_Y = k_fold(5, cut_sentences, label)
    train_X, train_Y, test_X, test_Y = k_fold(10, train_test_X[0], train_test_Y[0])
