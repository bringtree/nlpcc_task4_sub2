# 生成将数字转化字的功能。为测试集的句子服务，使得得以复原
import numpy as np
import os
import pickle

if __name__ == "__main__":
    # !!!有bug  intents_type_num 其实应该是11 但是

    train_args = {
        "embedding_words_num": 11863, "batch_size": 20, "time_step": 30, "sentences_num": 30, "intents_type_num": 12,
        "learning_rate": 0.0001, "hidden_num": 100, "enable_embedding": False, "iterations": 100
    }
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

        # 统计下个数
        # predict = model.get_result(sess, words_number_of_sentence, sentences_number_of_session, X_batch,
        #                            preprocessing_embedding_vec)

    with open("./result/sentence.txt", "w") as fp:
        result = [v + '\n' for v in result]
        fp.writelines(result)
