import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper
import numpy as np


class H_RNN():
    def __init__(self, embedding_words_num, batch_size, time_step, sentences_num, intents_type_num, learning_rate,
                 hidden_num,
                 enable_embedding):
        """

        :param embedding_words_num:  embedding能容纳词数量
        :param batch_size: batch 的容量
        :param time_step: 一个句子单词数量
        :param sentences_num: 一个session的句子数量
        :param intents_type_num: 意图数量+1
        :param learning_rate: 学习率
        :param hidden_num: lstm隐藏层数量
        :param enable_embedding: False为使用自己预先处理的词向量。True为使用模型自己来训练词向量
        """
        self.embedding_words_num = embedding_words_num
        self.time_step = time_step
        self.batch_size = batch_size
        self.sentences_num = sentences_num
        self.intents_type_num = intents_type_num
        self.learning_rate = learning_rate
        self.hidden_num = hidden_num
        self.enable_embedding = enable_embedding

    def build_model(self):
        self.output_keep_prob = tf.placeholder(shape=1, dtype=tf.float32)

        # 存放句子的数目 【batch_size】 也就是每个对话(session)中有的句子数目
        self.sentences_number_of_session = tf.placeholder(tf.int32, self.batch_size,
                                                          name="sentences_number_of_session")
        # 存放每个句子的单词个数 [句子数目，batch_size]
        self.words_number_of_sentence = tf.placeholder(tf.int32, [None, self.batch_size],
                                                       name="words_number_of_sentence")
        # [句子数目，句子长度，batch_size]
        self.encoder_inputs = tf.placeholder(tf.int32,
                                             [self.sentences_num, self.time_step, self.batch_size],
                                             name='encoder_inputs')
        if self.enable_embedding is True:
            # shape = self.batch_size, self.sentences_num_actual
            self.embedding = tf.get_variable(shape=[self.embedding_words_num, 300], dtype=tf.float32, name="embedding")
        else:
            self.embedding = tf.placeholder(shape=[self.embedding_words_num, 300], dtype=tf.float32, name="embedding")

        # shape = [session中的句子长度,句子中单词长度,batch_size大小,词向量长度]
        self.encoder_input_embeddings = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)

        encoder_f_cell_0 = LSTMCell(self.hidden_num, initializer=tf.orthogonal_initializer())
        encoder_b_cell_0 = LSTMCell(self.hidden_num, initializer=tf.orthogonal_initializer())

        encoder_f_cell_1 = DropoutWrapper(encoder_f_cell_0, output_keep_prob=self.output_keep_prob)
        encoder_b_cell_1 = DropoutWrapper(encoder_b_cell_0, output_keep_prob=self.output_keep_prob)

        def build_sentence_LSTM(input_embedding, encoder_inputs_actual_length):
            """
            构建底层单个句子的lstm
            其实 你要 注意的是 或者 你可以思考 我们写代码的时候 是站在 batch_size为1的角度去写 写完再修改成batch_size为你想要的大小。
            所以模型 在写的时候 batch_size 一般输入和输出中的shape都会带有batch_size这个维度。到哪 就跟到哪 。哪里都会有batch_size 的身影。
            然后 这个函数是这样子的 我们 单个句子lstm 输入的时候 无非就是要输入 单词。而单词在这里变成（有哪些单词。还有这个多少个单词(也就是句子长度)）
            所以这个LSTM 就是2个输入 1个是那些单词的输入（包括扩展0空白）和另外还有多少个单词。
            这样也就对应上面的 input_embedding shape [单词的数目(包括填充的),batch_size,词向量维度] 以及
            encoder_inputs_actual_length shape[batch_size] 每个batch_size 中的句子中 单词有多少个 注意 单个句子。我知道你疑惑上面 为什么不是多个句子
            我开头就说了 单个句子。 我后面会用map 让他 变成多个句子的。
            先写好单个句子 再循环复制。就变成多句子了。
            :param input_embedding:
            :param encoder_inputs_actual_length:
            :return:
            """
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

        # time_step,句子数目 ,batch_size,hidden_cell*2,    time_step,batch_size,hidden_cell*2
        # map 这样肯定 就要知道要循环多少次了。 1个session有多少个句子 就要循环多少次。很悲剧。这里我没法写成动态。全部都统一次数循环。
        # 循环次数 看下面 2个的长度(self.encoder_input_embeddings/self.words_number_of_sentence)的第一个维度。
        # 我也知道 你又在好奇x[0]和x[1]. 注意elems 我包了个小括号。他变成了个一个整体输入了。然后我知道你又会问 那为什么不只输入一个参数在在函数中去切。这样反而让你不好理解
        # 又是很悲催的告诉你。还记得上面循环第一个维度次数吧。 要是我两个合并在一起 同时作为1个参数输入，那就只循环2次。
        # 你就知道 我在这里纠结了多久了。
        all_encoder_outputs, all_encoder_final_state = tf.map_fn(fn=lambda x: build_sentence_LSTM(x[0], x[1]),
                                                                 elems=(self.encoder_input_embeddings,
                                                                        self.words_number_of_sentence),
                                                                 dtype=(tf.float32, tf.float32))

        with tf.variable_scope('top_encoder_layer'):
            top_cell_0 = LSTMCell(self.hidden_num, initializer=tf.orthogonal_initializer())
            top_cell_1 = DropoutWrapper(top_cell_0, output_keep_prob=self.output_keep_prob)
            # 【time_step，batch_size，hidden_cell】
            (top_outputs), (top_fw_final_state) \
                = tf.nn.dynamic_rnn(cell=top_cell_1,
                                    inputs=all_encoder_final_state,
                                    sequence_length=self.sentences_number_of_session,
                                    dtype=tf.float32,
                                    time_major=True)
        # [time_step,batch_size,hidden_num*2]
        self.top_outputs = top_outputs
        # [barch_size,hidden_num,hidden_num*2]
        self.top_outputs = tf.transpose(self.top_outputs, perm=[1, 0, 2])

        # 真实的标签 。这个只是为了做计算 计算loss。需要有个真实的标签和预测的标签来计算loss。
        self.the_true_inputs = tf.placeholder(shape=[self.batch_size, None], dtype=tf.int32, name="the_true_inputs")

        # 顶层输出。为什么要reshape。因为没有3维的乘法(不能说没有吧 可以自己造个乘法 比如卷积?_?)。所以要降维成2维矩阵。才有矩阵乘法
        self.top_outputs = tf.reshape(self.top_outputs, [-1, self.hidden_num])

        # 参数W
        intent_W = tf.get_variable(
            initializer=tf.random_uniform([self.hidden_num, self.intents_type_num], -0.1, 0.1),
            dtype=tf.float32, name="intent_W")
        # 参数B
        intent_b = tf.get_variable(initializer=tf.zeros([self.intents_type_num]), dtype=tf.float32, name="intent_b")
        # W*x
        self.top_outputs = tf.matmul(self.top_outputs, intent_W)
        # +b
        self.top_outputs = tf.add(self.top_outputs, intent_b)
        # 把二维变回3维
        self.top_outputs = tf.reshape(self.top_outputs, [self.batch_size, self.sentences_num, self.intents_type_num])

        # 把输出的结果 转为概率
        self.predict = tf.sigmoid(self.top_outputs)
        # 求概率最大的那个的类的标号
        self.predict = tf.argmax(self.predict, axis=2)

        # 计算哪些是空白输入 哪些是真实有输入的。 具体用法见这个
        # https://github.com/bringtree/everydayCommond/issues/77
        self.mask = tf.to_float(tf.not_equal(self.the_true_inputs, 0))

        regularization_cost = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.top_outputs,
                                                     targets=self.the_true_inputs,
                                                     weights=self.mask) + regularization_cost
        # 梯度函数
        optimizer = tf.train.AdamOptimizer(name="a_optimizer", learning_rate=self.learning_rate)
        self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
        self.gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
        self.train_op = optimizer.apply_gradients(zip(self.gradients, self.vars))

    def train(self, sess, words_number_of_sentence, sentences_number_of_session, encoder_inputs, the_true_inputs,
              train_output_keep_prob,embedding_W=None):
        """

        :param sess:
        :param words_number_of_sentence:  存放每个句子的单词个数 [句子数目，batch_size]
        :param sentences_number_of_session:  存放句子的数目 【batch_size】 也就是每个对话(session)中有的句子数目
        :param encoder_inputs: 所有输入。填充要做上。[句子数目，句子长度，batch_size]
        :param the_true_inputs: 真实的输出结果，填充要做上
        :param embedding_W: None为使用模型自己训练的词向量。不让自己要带上。shape [self.embedding_words_num, 300]
        :return: 返回loss大小。忘记是个值 还是一个矩阵了。。。 看api 看上去 应该是个值吧。但是 交叉商一般返回是个矩阵。自己试一试吧。我机子一个batch_size都跑不动。
        """
        if self.enable_embedding is True:
            loss = sess.run([self.loss, self.train_op], feed_dict={
                self.words_number_of_sentence: words_number_of_sentence,
                self.sentences_number_of_session: sentences_number_of_session,
                self.encoder_inputs: encoder_inputs,
                self.the_true_inputs: the_true_inputs,
                self.output_keep_prob: train_output_keep_prob
            })
        else:
            loss = sess.run([self.loss, self.train_op], feed_dict={
                self.words_number_of_sentence: words_number_of_sentence,
                self.sentences_number_of_session: sentences_number_of_session,
                self.encoder_inputs: encoder_inputs,
                self.the_true_inputs: the_true_inputs,
                self.embedding: embedding_W,
                self.output_keep_prob: train_output_keep_prob
            })
        return loss

    def get_result(self, sess, words_number_of_sentence, sentences_number_of_session, encoder_inputs,
                   test_output_keep_prob, embedding_W=None):
        """
        返回batch_size个大小的30个句子的意图输出 shape: [batch_size,sentences_num,intents_type_num]
        :param sess:
        :param words_number_of_sentence:存放每个句子的单词个数 [句子数目，batch_size]
        :param sentences_number_of_session:  存放句子的数目 【batch_size】 也就是每个对话(sess
        :param encoder_inputs:所有输入。填充要做上。[句子数目，句子长度，batch_size]
        :return: 返回一个预测的结果结果。然后 你要自己拉到外面去比对(注意带有填充0的)。
        """
        if self.enable_embedding is True:
            predict_result = sess.run(self.predict, feed_dict={
                self.words_number_of_sentence: words_number_of_sentence,
                self.sentences_number_of_session: sentences_number_of_session,
                self.encoder_inputs: encoder_inputs,
                self.output_keep_prob: test_output_keep_prob
            })
            return predict_result
        else:
            predict_result = sess.run(self.predict, feed_dict={
                self.words_number_of_sentence: words_number_of_sentence,
                self.sentences_number_of_session: sentences_number_of_session,
                self.encoder_inputs: encoder_inputs,
                self.embedding: embedding_W,
                self.output_keep_prob: test_output_keep_prob

            })
            return predict_result
