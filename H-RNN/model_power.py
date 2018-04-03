import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper
import numpy as np


class H_RNN():
    def __init__(self, word_num, batch_size, time_step, sentences_num, intents_type_num, learning_rate, hidden_num):
        self.word_num = word_num
        self.time_step = time_step
        self.batch_size = batch_size
        self.sentences_num = sentences_num
        self.intents_type_num = intents_type_num
        self.learning_rate = learning_rate
        self.hidden_num = hidden_num

    def build_model(self):
        # 存放句子的数目 【batch_size】
        self.sentences_inputs_actual_num = tf.placeholder(tf.int32, self.batch_size,
                                                          name="sentences_inputs_actual_num")
        # 存放每个句子的单词个数 [句子数目，batch_size]

        self.word_inputs_actual_length = tf.placeholder(tf.int32, [None, self.batch_size],
                                                        name="word_inputs_actual_length")

        # [句子数目，句子长度，batch_size]
        self.encoder_inputs = tf.placeholder(tf.int32,
                                             [self.sentences_num, self.time_step, self.batch_size],
                                             name='encoder_inputs')

        # shape = self.batch_size, self.sentences_num_actual
        self.embedding = tf.get_variable(shape=[self.word_num, 400], dtype=tf.float32, name="embedding")
        encoder_input_embeddings = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)

        encoder_f_cell_0 = LSTMCell(self.hidden_num, initializer=tf.orthogonal_initializer())
        encoder_b_cell_0 = LSTMCell(self.hidden_num, initializer=tf.orthogonal_initializer())

        encoder_f_cell_1 = DropoutWrapper(encoder_f_cell_0, output_keep_prob=0.5)
        encoder_b_cell_1 = DropoutWrapper(encoder_b_cell_0, output_keep_prob=0.5)

        def build_sentence_LSTM(input_embedding, encoder_inputs_actual_length):
            # time_major=True时 input输入为[time_step, batch_size, embedding_size]
            (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) \
                = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_f_cell_1,
                                                  cell_bw=encoder_b_cell_1,
                                                  inputs=input_embedding,
                                                  sequence_length=encoder_inputs_actual_length,
                                                  dtype=tf.float32,
                                                  time_major=True)
            # shape  = [time_step, batch_size, embedding_size, (hidden_size)100*2]
            encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
            encoder_final_state = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h,), 1)
            return encoder_outputs, encoder_final_state

        # 7 ,30 ,128,200,    7,128,200
        all_encoder_outputs, all_encoder_final_state = tf.map_fn(fn=lambda x: build_sentence_LSTM(x[0], x[1]),
                                                                 elems=(encoder_input_embeddings,
                                                                        self.word_inputs_actual_length),
                                                                 dtype=(tf.float32, tf.float32))

        with tf.variable_scope('top_encoder_layer'):
            top_f_cell_0 = LSTMCell(self.hidden_num, initializer=tf.orthogonal_initializer())
            top_b_cell_0 = LSTMCell(self.hidden_num, initializer=tf.orthogonal_initializer())

            top_f_cell_1 = DropoutWrapper(top_f_cell_0, output_keep_prob=0.5)
            top_b_cell_1 = DropoutWrapper(top_b_cell_0, output_keep_prob=0.5)
            # 【7，128，100】
            (top_fw_outputs, top_bw_outputs), (top_fw_final_state, top_bw_final_state) \
                = tf.nn.bidirectional_dynamic_rnn(cell_fw=top_f_cell_1,
                                                  cell_bw=top_b_cell_1,
                                                  inputs=all_encoder_final_state,
                                                  sequence_length=self.sentences_inputs_actual_num,
                                                  dtype=tf.float32,
                                                  time_major=True)
        # [7,128,200]
        self.top_outputs = tf.concat((top_fw_outputs, top_bw_outputs), 2)
        # [128,7,200]
        self.top_outputs = tf.transpose(self.top_outputs, perm=[1, 0, 2])

        # intent
        self.the_true_inputs = tf.placeholder(shape=[self.batch_size, None], dtype=tf.int32, name="the_true_inputs")

        self.top_outputs = tf.reshape(self.top_outputs, [-1, self.hidden_num * 2])

        intent_W = tf.get_variable(
            initializer=tf.random_uniform([self.hidden_num * 2, self.intents_type_num], -0.1, 0.1),
            dtype=tf.float32, name="intent_W")
        intent_b = tf.get_variable(initializer=tf.zeros([self.intents_type_num]), dtype=tf.float32, name="intent_b")
        self.top_outputs = tf.matmul(self.top_outputs, intent_W)
        self.top_outputs = tf.add(self.top_outputs, intent_b)
        self.top_outputs = tf.reshape(self.top_outputs, [self.batch_size, self.sentences_num, self.intents_type_num])

        self.predict = tf.sigmoid(self.top_outputs)
        self.predict = tf.argmax(self.predict, axis=2)
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.top_outputs,
                                                            labels=tf.one_hot(self.the_true_inputs,
                                                                              self.intents_type_num))

        optimizer = tf.train.AdamOptimizer(name="a_optimizer", learning_rate=self.learning_rate)
        self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))

        # print("vars for loss function: ", self.vars)
        self.gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
        self.train_op = optimizer.apply_gradients(zip(self.gradients, self.vars))

    def train(self, sess, word_inputs_actual_length, sentences_inputs_actual_num, encoder_inputs, the_true_inputs):
        loss = sess.run([self.loss, self.train_op], feed_dict={
            model.word_inputs_actual_length: word_inputs_actual_length,
            model.sentences_inputs_actual_num: sentences_inputs_actual_num,
            model.encoder_inputs: encoder_inputs,
            model.the_true_inputs: the_true_inputs
        })
        return loss

    def get_result(self, sess, word_inputs_actual_length, sentences_inputs_actual_num, encoder_inputs):
        """
        返回batch_size个大小的30个句子的意图输出 shape: [batch_size,sentences_num,intents_type_num]
        :param sess:
        :param word_inputs_actual_length:
        :param sentences_inputs_actual_num:
        :param encoder_inputs:
        :return:
        """
        predict_result = sess.run(self.predict, feed_dict={
            model.word_inputs_actual_length: word_inputs_actual_length,
            model.sentences_inputs_actual_num: sentences_inputs_actual_num,
            model.encoder_inputs: encoder_inputs,
        })
        return predict_result


if __name__ == "__main__":
    # model = H_RNN(word_num=11916, batch_size=64, time_step=20, sentences_num=15)
    model = H_RNN(word_num=13000, batch_size=20, time_step=30, sentences_num=30, intents_type_num=11,
                  learning_rate=0.001, hidden_num=100)
    model.build_model()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    sentences_inputs = np.load("./data/new_paragraph_sentences.npy")
    labels_inputs = np.load("./data/pad_new_paragraph_labels.npy")

    X_batches = []
    Y_batches = []
    begin_index = 0
    end_index = 20
    while end_index < len(sentences_inputs):
        X_batches.append(sentences_inputs[begin_index:end_index])
        Y_batches.append(labels_inputs[begin_index:end_index])
        begin_index = end_index
        end_index = end_index + 20


    def non_zero_times_count(sentence):
        num = 0
        for v in sentence:
            if v != 0:
                num += 1
        return num


    batch_size = len(X_batches)
    for epoch in range(50):
        correct_num = 0
        mistake_num = 0
        for batches_times in range(batch_size):
            X_batch = X_batches[batches_times]
            Y_batch = Y_batches[batches_times]

            # 存放每个句子的单词个数 shape:[batch_size(20)，句子数目(30)]
            word_inputs_actual_length = []
            for paragraph_idx, paragraph in enumerate(X_batch):
                # 每个wordinputlength 中要放入 15个句子的单词个数
                tmp = np.zeros(30, dtype="int32")
                for sentence_idx, sentence in enumerate(paragraph):
                    tmp[sentence_idx] = non_zero_times_count(sentence)
                word_inputs_actual_length.append(tmp)
            word_inputs_actual_length = np.array(word_inputs_actual_length)

            # 存放句子的数目 shape【batch_size】
            sentences_inputs_actual_num = [non_zero_times_count(v) for v in word_inputs_actual_length]
            # 转成格式    shape:[句子数目(15)，batch_size(20)]
            word_inputs_actual_length = np.transpose(word_inputs_actual_length, [1, 0])
            X_batch = np.transpose(X_batch, [1, 2, 0])

            loss, _ = model.train(sess, word_inputs_actual_length, sentences_inputs_actual_num, X_batch, Y_batch)
            predict = model.get_result(sess, word_inputs_actual_length, sentences_inputs_actual_num, X_batch)

            for batch_size_idx, batch_size_content in enumerate(Y_batch):
                for sentence_idx, sentence_intent in enumerate(batch_size_content):
                    if sentence_intent == 0:
                        break
                    else:
                        if sentence_intent == predict[batch_size_idx][sentence_idx]:
                            correct_num += 1
                        else:
                            mistake_num += 1

        print("acc:" + str(correct_num / (mistake_num + correct_num)))

    # loss, _ = model.train(sess, word_inputs_actual_length, sentences_inputs_actual_num, X_batch, Y_batch)
    # if batches_times == 0:
    #     print(loss.mean())
