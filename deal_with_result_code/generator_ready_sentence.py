# 生成将数字转化字的功能。为测试集的句子服务，使得得以复原
import numpy as np
import os
import pickle
os.chdir("../")
from config import train_args

if __name__ == "__main__":
    # 数据集的序号 k_fold_index
    # 模型保存地址
    result = []
    # 加载句子的字典
    with open("./data/word_dict_reverse.pkl", "rb") as fp:
        word_dict_reverse = pickle.load(fp)

    # 数据加载
    test_X = np.load("./10_fold_corpus/test_X.npy")
    test_Y = np.load("./10_fold_corpus/test_Y.npy")


    def non_zero_times_count(sentence):
        """
        统计句子中多少个不是0（也就是有多少个单词）
        :param sentence:
        :return:
        """
        num = 0
        for v in sentence:
            if v != 0:
                num += 1
        return num


    test_X = np.concatenate((test_X,
                             np.zeros(shape=(train_args["batch_size"] - len(test_X) % train_args["batch_size"],
                                             train_args["sentences_num"], train_args["time_step"]),
                                      dtype=np.int32)), axis=0)
    test_Y = np.concatenate((test_Y,
                             np.zeros(shape=(train_args["batch_size"] - len(test_Y) % train_args["batch_size"],
                                             train_args["sentences_num"]),
                                      dtype=np.int32)), axis=0)
    test_X_batches = []
    test_begin_index = 0
    test_end_index = train_args['batch_size']
    while test_end_index < len(test_X):
        test_X_batches.append(test_X[test_begin_index:test_end_index])
        test_begin_index = test_end_index
        test_end_index = test_end_index + train_args['batch_size']

    for X_batch in test_X_batches:
        for paragraph in X_batch:
            for sentence in paragraph:
                tmp = [word_dict_reverse[v] for v in sentence if v != 0]
                if tmp != []:
                    result.append("".join(tmp))



    with open("./result/sentence.txt", "w") as fp:
        result = [v + '\n' for v in result]
        fp.writelines(result)
