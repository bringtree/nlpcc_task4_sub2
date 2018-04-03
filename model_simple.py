import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, DropoutWrapper
import numpy as np


class H_RNN():
    def __init__(self, word_num, batch_size, time_step, sentences_num):
        self.word_num = word_num
        self.time_step = time_step
        self.batch_size = batch_size
        self.sentences_num = sentences_num

    def build_model(self):
        # 存放句子的数目 【batch_size】
        self.sentences_inputs_actual_num = tf.placeholder(tf.int32, self.batch_size,
                                                          name="sentences_inputs_actual_num")
        # 存放每个句子的单词个数 [句子数目，batch_size]

        self.word_inputs_actual_length = tf.placeholder(tf.int32, [None, self.batch_size],
                                                        name="word_inputs_actual_length")

        # [句子数目，句子长度，batch_size]
        # word_num=120, batch_size=182,  time_step=30, sentences_num=7
        self.encoder_inputs = tf.placeholder(tf.int32,
                                             [self.sentences_num, self.time_step, self.batch_size],
                                             name='encoder_inputs')

        # shape = self.batch_size, self.sentences_num_actual
        self.embedding = tf.get_variable(shape=[self.word_num, 400], dtype=tf.float32, name="embedding")
        encoder_input_embeddings = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)

        encoder_f_cell_0 = LSTMCell(100, initializer=tf.orthogonal_initializer())
        encoder_b_cell_0 = LSTMCell(100, initializer=tf.orthogonal_initializer())

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
            top_f_cell_0 = LSTMCell(100, initializer=tf.orthogonal_initializer())
            top_b_cell_0 = LSTMCell(100, initializer=tf.orthogonal_initializer())

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

        self.top_outputs = tf.reshape(self.top_outputs, [-1, 100 * 2])

        intent_W = tf.get_variable(
            initializer=tf.random_uniform([100 * 2, 3], -0.1, 0.1),
            dtype=tf.float32, name="intent_W")
        intent_b = tf.get_variable(initializer=tf.zeros([3]), dtype=tf.float32, name="intent_b")
        self.top_outputs = tf.matmul(self.top_outputs, intent_W)
        self.top_outputs = tf.add(self.top_outputs, intent_b)
        self.top_outputs = tf.reshape(self.top_outputs, [self.batch_size, self.sentences_num, 3])
        # self.top_outputs = tf.sigmoid(self.top_outputs)
        self.predict = tf.sigmoid(self.top_outputs)
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.top_outputs,
                                                            labels=tf.one_hot(self.the_true_inputs, 3))

        optimizer = tf.train.AdamOptimizer(name="a_optimizer", learning_rate=0.001)
        self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))

        # print("vars for loss function: ", self.vars)
        self.gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
        self.train_op = optimizer.apply_gradients(zip(self.gradients, self.vars))

    def train(self, sess, word_inputs_actual_length, sentences_inputs_actual_num, encoder_inputs, the_true_inputs):
        loss = sess.run([model.loss, model.train_op], feed_dict={
            model.word_inputs_actual_length: word_inputs_actual_length,
            model.sentences_inputs_actual_num: sentences_inputs_actual_num,
            model.encoder_inputs: encoder_inputs,
            model.the_true_inputs: the_true_inputs
        })
        return loss


if __name__ == "__main__":
    # model = H_RNN(word_num=11916, batch_size=64, time_step=20, sentences_num=15)
    model = H_RNN(word_num=80, batch_size=2, time_step=8, sentences_num=7)
    model.build_model()
    sess = tf.Session()
    # writer = tf.summary.FileWriter("./model_graph")
    # writer.add_graph(sess.graph)
    pad_new_paragraph_labels = np.load("pad_new_paragraph_labels.npy")
    new_paragraph_sentences = np.load("new_paragraph_sentences.npy")

    sess.run(tf.global_variables_initializer())
    # test_input = np.array([
    #     [
    #         [1, 2, 3, 4, 5, 6, 7, 8],
    #         [1, 4, 9, 16, 25, 36, 49, 64],
    #         [2, 3, 4, 5, 6, 7, 8, 9],
    #         [2, 3, 4, 5, 6, 7, 8, 9],
    #         [3, 4, 5, 6, 7, 8, 9, 1],
    #         [4, 5, 6, 7, 8, 9, 10, 11],
    #         [5, 6, 7, 8, 9, 10, 11, 12]
    #     ],
    #     [
    #         [1, 2, 3, 4, 5, 0, 0, 0],
    #         [1, 4, 9, 16, 25, 36, 0, 0],
    #         [2, 3, 4, 5, 6, 7, 8, 0],
    #         [2, 3, 4, 5, 6, 7, 8, 9],
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #     ]
    # ])
    test_input = np.array(new_paragraph_sentences)
    test_input = np.transpose(test_input, [1, 2, 0])

    # 存放每个句子的单词个数 shape [句子数目，batch_size]
    # word_inputs_actual_length = np.array([
    #     [8, 8, 8, 8, 8, 8, 8],
    #     [5, 6, 7, 8, 0, 0, 0]
    # ])
    # word_inputs_actual_length = np.transpose(word_inputs_actual_length, [1, 0])

    word_inputs_actual_length = np.array([
        [8, 8, 8, 8, 8, 8, 8],
        [5, 6, 7, 8, 0, 0, 0]
    ])
    word_inputs_actual_length = np.transpose(word_inputs_actual_length, [1, 0])


    # 存放句子的数目 shape【batch_size】
    sentences_inputs_actual_num = np.array([7, 4])

    # test_intent = np.array([[1, 1, 2, 2, 2, 2, 2],
    #                         [1, 1, 2, 2, 0, 0, 0]])
    test_intent = np.array(pad_new_paragraph_labels)
    print(sess.run(model.predict, feed_dict={
        model.word_inputs_actual_length: word_inputs_actual_length,
        model.sentences_inputs_actual_num: sentences_inputs_actual_num,
        model.encoder_inputs: test_input,
        model.the_true_inputs: test_intent
    }))
    for v in range(1000):
        loss, _ = model.train(sess, word_inputs_actual_length, sentences_inputs_actual_num, test_input, test_intent)

    print(sess.run(model.predict, feed_dict={
        model.word_inputs_actual_length: word_inputs_actual_length,
        model.sentences_inputs_actual_num: sentences_inputs_actual_num,
        model.encoder_inputs: test_input,
        model.the_true_inputs: test_intent
    }))
