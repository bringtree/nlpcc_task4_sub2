from model_hrnn import H_RNN
import tensorflow as tf
import numpy as np
import pickle
import os

if __name__ == "__main__":
    # !!!有bug  intents_type_num 其实应该是11 但是
    train_args = {
        "embedding_words_num": 11863, "batch_size": 20, "time_step": 30, "sentences_num": 30, "intents_type_num": 12,
        "learning_rate": 0.0001, "hidden_num": 100, "enable_embedding": False, "iterations": 100
    }
    # 数据集的序号 k_fold_index
    # 模型保存地址
    for k_fold_index in range(10):
        # 模型的路径
        model_src = './save_model/k_fold_index' + str(k_fold_index) + '/'
        # 数据加载
        # test_X = np.load("./10_fold_corpus/test_X_data_" + str(k_fold_index) + ".npy")
        test_X = np.load("./10_fold_corpus/test_X.npy")
        # 输出的结果
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

        tf.reset_default_graph()

        model = H_RNN(
            embedding_words_num=train_args["embedding_words_num"],
            batch_size=train_args["batch_size"],
            time_step=train_args["time_step"],
            sentences_num=train_args["sentences_num"],
            intents_type_num=train_args["intents_type_num"],
            learning_rate=train_args["learning_rate"],
            hidden_num=train_args["hidden_num"],
            enable_embedding=train_args["enable_embedding"]
        )

        model.build_model()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # 加载模型
        ckpt = tf.train.get_checkpoint_state(model_src)
        saver.restore(sess, os.path.join(ckpt.model_checkpoint_path))


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


        if train_args["enable_embedding"] == False:
            # self.embedding = tf.placeholder(shape=[self.embedding_words_num, 300], dtype=tf.float32, name="embedding")
            with open("./data/vec_dict.pkl", 'rb') as fp:
                vec_dict = pickle.load(fp)
                # 没有0的
                preprocessing_embedding_vec = np.eye(len(vec_dict) + 1, 300)
                for key, value in vec_dict.items():
                    preprocessing_embedding_vec[key] = value

        test_X_batches = []
        test_Y_batches = []
        test_begin_index = 0
        test_end_index = train_args['batch_size']
        while test_end_index < len(test_X):
            test_X_batches.append(test_X[test_begin_index:test_end_index])
            test_begin_index = test_end_index
            test_end_index = test_end_index + train_args['batch_size']

        test_batch_size = len(test_X_batches)

        for epoch in range(1):

            for batches_times in range(test_batch_size):
                X_batch = test_X_batches[batches_times]

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

                # 统计下个数
                predict = model.get_result(sess, words_number_of_sentence, sentences_number_of_session, X_batch,
                                           preprocessing_embedding_vec)

                for paragraph_idx, sentence_num in enumerate(sentences_number_of_session):
                    for i in range(sentence_num):
                        result.append(labels_dict_reverse[str(predict[paragraph_idx][i])])

            with open("./result/result" + str(k_fold_index) + ".txt", "w") as fp:
                result = [v + '\n' for v in result]
                fp.writelines(result)
