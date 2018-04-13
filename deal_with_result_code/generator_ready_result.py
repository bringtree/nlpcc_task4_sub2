# 生成将数字转化字的功能。为测试集的意图服务，使得得以复原
import numpy as np
import os
os.chdir("../")

if __name__ == "__main__":
    # !!!有bug  intents_type_num 其实应该是11 但是

    train_args = {
        "embedding_words_num": 11863, "vec_size": 400, "batch_size": 20, "time_step": 30, "sentences_num": 30,
        "intents_type_num": 12, "learning_rate": 0.0001, "hidden_num": 100, "enable_embedding": False,
        "iterations": 100, "train_output_keep_prob": 0.5, "test_output_keep_prob": 1
    }
    # 数据集的序号 k_fold_index
    # 模型保存地址
    result = []
    # 加载意图标签的字典
    labels_dict = {}
    with open("./data/labels.txt") as fp:
        labels_type = [v[:-1] for v in fp.readlines()]
    label_dict = {}
    i = 1
    for v in labels_type:
        label_dict[v] = i
        i += 1
    labels_dict_reverse = {}
    for key, value in label_dict.items():
        labels_dict_reverse[str(value)] = key

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
    test_Y_batches = []
    test_begin_index = 0
    test_end_index = train_args['batch_size']
    while test_end_index < len(test_X):
        test_X_batches.append(test_X[test_begin_index:test_end_index])
        test_Y_batches.append(test_Y[test_begin_index:test_end_index])
        test_begin_index = test_end_index
        test_end_index = test_end_index + train_args['batch_size']

    test_batch_size = len(test_X_batches)

    for batches_times in range(test_batch_size):
        X_batch = test_X_batches[batches_times]
        Y_batch = test_Y_batches[batches_times]

        # 存放每个句子的单词个数 shape:[batch_size(20)，句子数目(30)]
        words_number_of_sentence = []
        for paragraph_idx, paragraph in enumerate(X_batch):
            # 每个words_number_of_sentence 中要放入 30个句子的单词个数
            tmp = np.zeros(train_args["sentences_num"], dtype="int32")
            for sentence_idx, sentence in enumerate(paragraph):
                tmp[sentence_idx] = non_zero_times_count(sentence)
            words_number_of_sentence.append(tmp)
        words_number_of_sentence = np.array(words_number_of_sentence)

        # 存放句子的数目 shape[batch_size]
        sentences_number_of_session = [non_zero_times_count(v) for v in words_number_of_sentence]
        # 转成格式    shape:[句子数目(30)，batch_size(20)]
        words_number_of_sentence = np.transpose(words_number_of_sentence, [1, 0])
        X_batch = np.transpose(X_batch, [1, 2, 0])

        for batch_size_idx, batch_size_content in enumerate(Y_batch):
            for sentence_idx, sentence_intent in enumerate(batch_size_content):
                if sentence_intent == 0:
                    break
                else:
                    result.append(labels_dict_reverse[str(sentence_intent)])

    with open("./result/resulttrue.txt", "w") as fp:
        result = [v + '\n' for v in result]
        fp.writelines(result)
