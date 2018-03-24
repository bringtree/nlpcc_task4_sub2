# coding=utf-8
# @author: cer
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, DropoutWrapper
import sys


class Model:
    def __init__(self, input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, batch_size=16, embedding_w=None):
        self.input_steps = input_steps
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.epoch_num = epoch_num
        if embedding_w is None:
            tf.random_uniform([self.vocab_size, self.embedding_size],
                              -0.1, 0.1)
        else:
            self.embedding_W = embedding_w
        # 每句输入的实际长度，除了padding
        self.encoder_inputs_actual_length = tf.placeholder(tf.int32, [batch_size],
                                                           name='encoder_inputs_actual_length')
        self.decoder_targets = tf.placeholder(tf.int32, [batch_size, input_steps],
                                              name='decoder_targets')
        # 真实的intent输入
        self.intent_targets = tf.placeholder(tf.int32, [batch_size],
                                             name='intent_targets')

    def build(self):
        with tf.name_scope('embedding_layer'):
            self.encoder_inputs = tf.placeholder(tf.int32, [self.input_steps, self.batch_size],
                                                 name='encoder_inputs')
            self.embeddings = tf.Variable(self.embedding_W, dtype=tf.float32, name="embedding")

            # self.embeddings
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
            # <tf.Tensor 'embedding_layer/embedding_lookup:0' shape=(30, 64, 100) dtype=float32>

        with tf.name_scope('encoder_layer'):
            # 使用单个LSTM cell
            encoder_f_cell_0 = LSTMCell(self.hidden_size, initializer=tf.orthogonal_initializer())
            encoder_b_cell_0 = LSTMCell(self.hidden_size, initializer=tf.orthogonal_initializer())
            # output_keep_prob :if it is constant and 1, no output dropout will be added.
            encoder_f_cell = DropoutWrapper(encoder_f_cell_0, output_keep_prob=0.5)
            encoder_b_cell = DropoutWrapper(encoder_b_cell_0, output_keep_prob=0.5)
            # encoder_inputs_time_major = tf.transpose(self.encoder_inputs_embedded, perm=[1, 0, 2])
            # 下面四个变量的尺寸：T*B*D，T*B*D，B*D，B*D
            #  Time_step, Batch_size, Hidden_size
            # Time_major决定了inputs Tensor前两个dim表示的含义
            # time_major=False时[batch_size, sequence_length, embedding_size]
            # time_major=True时[sequence_length, batch_size, embedding_size]
            (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_f_cell,
                                                cell_bw=encoder_b_cell,
                                                inputs=self.encoder_inputs_embedded,
                                                sequence_length=self.encoder_inputs_actual_length,
                                                dtype=tf.float32, time_major=True)

            # 每个句子的单词数量* 句子数量 * LSTMCell 的 num_units的长度 *2
            encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            # ct = forget_gate_probability * c(t-1) + input_gate_probability * input  就是State
            # 句子的数量 * LSTMCell 的 num_units的长度 *2
            # encoder_final_state_c = tf.concat(
            #     (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
            # # ht = ct * output_gate_probability 就是输出
            # # LSTMCell 的 num_units的长度 *2
            encoder_final_state_h = tf.concat(
                (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

            # 一种数据结构 LSTMStateTuple(c=<tf.Tensor 'concat_1:0' shape=(16, 200) dtype=float32>, h=<tf.Tensor 'concat_2:0' shape=(16, 200) dtype=float32>)
            # self.encoder_final_state = LSTMStateTuple(
            #     c=encoder_final_state_c,
            #     h=encoder_final_state_h
            # )
            intent_W = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.intent_size], -0.1, 0.1),
                                   dtype=tf.float32, name="intent_W")
            intent_b = tf.Variable(tf.zeros([self.intent_size]), dtype=tf.float32, name="intent_b")

            # 求intent [句子数量,cell_num*2(concat)] + intene_W(cell_num*2,intent_size) + b
            intent_logits = tf.add(tf.matmul(encoder_final_state_h, intent_W), intent_b)
            # intent_prob = tf.nn.softmax(intent_logits)
            self.intent = tf.argmax(intent_logits, axis=1)

        with tf.name_scope('decoder_layer'):
            decoder_lengths = self.encoder_inputs_actual_length

            with tf.name_scope('helper_function'):
                # 这块开始就出现helper的代码了 完全是迷
                # self.slot_W = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.slot_size], -1, 1),
                #                           dtype=tf.float32, name="slot_W")
                # self.slot_b = tf.Variable(tf.zeros([self.slot_size]), dtype=tf.float32, name="slot_b")
                sos_time_slice = tf.ones([self.batch_size], dtype=tf.int32, name='SOS') * 2
                sos_step_embedded = tf.nn.embedding_lookup(self.embeddings, sos_time_slice)
                # pad_time_slice = tf.zeros([self.batch_size], dtype=tf.int32, name='PAD')
                # pad_step_embedded = tf.nn.embedding_lookup(self.embeddings, pad_time_slice)
                pad_step_embedded = tf.zeros([self.batch_size, self.hidden_size * 2 + self.embedding_size],
                                             dtype=tf.float32)

                def initial_fn():
                    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
                    initial_input = tf.concat((sos_step_embedded, encoder_outputs[0]), 1)
                    return initial_elements_finished, initial_input

                def sample_fn(time, outputs, state):
                    # 选择logit最大的下标作为sample
                    print("outputs", outputs)
                    # output_logits = tf.add(tf.matmul(outputs, self.slot_W), self.slot_b)
                    # print("slot output_logits: ", output_logits)
                    # prediction_id = tf.argmax(output_logits, axis=1)
                    #
                    # argmax 返回最大值的 x参数
                    # array([[1, 2, 3, 4],
                    #        [5, 6, 7, 8]])
                    # >> > e = tf.argmax(a, 1)
                    # >> > sess.run(e)
                    # array([3, 3])
                    prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
                    return prediction_id

                def next_inputs_fn(time, outputs, state, sample_ids):
                    # 上一个时间节点上的输出类别，获取embedding再作为下一个时间节点的输入
                    pred_embedding = tf.nn.embedding_lookup(self.embeddings, sample_ids)
                    # 输入是h_i+o_{i-1}+c_i
                    next_input = tf.concat((pred_embedding, encoder_outputs[time]), 1)
                    elements_finished = (
                        time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
                    all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
                    next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
                    next_state = state
                    return elements_finished, next_inputs, next_state

                my_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

                # 转化下shape TensorShape([Dimension(16), Dimension(50), Dimension(200)])

            with tf.name_scope('attention'):
                memory = tf.transpose(encoder_outputs, [1, 0, 2])
                # Bahdanau 加法注意力 Luong 乘法注意力
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.hidden_size, memory=memory,
                    memory_sequence_length=self.encoder_inputs_actual_length)
                cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size * 2, initializer=tf.orthogonal_initializer())

                # 注意力包装器 需要 cell 和 注意力机制
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell, attention_mechanism, attention_layer_size=self.hidden_size)
                # 这里 要不要 加一层 dropout？
                # attn_cell = tf.contrib.rnn.DropoutWrapper(attn_cell, output_keep_prob=0.5)
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attn_cell, self.slot_size, reuse=None
                )

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=my_helper,
                initial_state=out_cell.zero_state(
                    dtype=tf.float32, batch_size=self.batch_size))

            # initial_state=encoder_final_state)
            # final_outputs =[shape=(?, 16, 122),shape=(?, 16)]
            # final_state = [shape=(16, 200),shape=(16, 100) ]
            # final_sequence_lengths = [shape=(16,)
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=True,
                impute_finished=True, maximum_iterations=self.input_steps
            )
            outputs = final_outputs


        # 这个就是 槽输出
        self.decoder_prediction = outputs.sample_id
        # max_step对应的是slot输出 batch_size 是句子 dim是输出
        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(outputs.rnn_output))
        self.decoder_targets_time_majored = tf.transpose(self.decoder_targets, [1, 0])
        self.decoder_targets_true_length = self.decoder_targets_time_majored[:decoder_max_steps]
        print("decoder_targets_true_length: ", self.decoder_targets_true_length)

        # 定义mask，使padding不计入loss计算
        self.mask = tf.to_float(tf.not_equal(self.decoder_targets_true_length, 0))
        # 定义slot标注的损失
        with tf.name_scope("lose_function"):
            loss_slot = tf.contrib.seq2seq.sequence_loss(
                outputs.rnn_output, self.decoder_targets_true_length, weights=self.mask)

            # 定义intent分类的损失
            # 我把损失函数换成了 sigmoid 原来是softmax 可以比较下
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.one_hot(self.intent_targets, depth=self.intent_size, dtype=tf.float32),
                logits=intent_logits)
            loss_intent = tf.reduce_mean(cross_entropy)

            # 我觉得 其实应该把2个loss 分出来 单独训练 或者加权？猜测 前期intent正确率不行 后期slot 不行
            self.loss = loss_slot + loss_intent
            # tf.summary.scalar('loss_slot', loss_slot)
            # tf.summary.scalar('loss_intent', loss_intent)
            # tf.summary.scalar('all_loss', self.loss)

        with tf.name_scope("optimizer_function"):
            optimizer = tf.train.AdamOptimizer(name="a_optimizer", learning_rate=0.001)
            self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
            # print("vars for loss function: ", self.vars)
            self.gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
            self.train_op = optimizer.apply_gradients(zip(self.gradients, self.vars))

        with tf.name_scope("accuracy"):
            self.intent_accs_placeholder = tf.placeholder(shape=[None], dtype=tf.float32)
            self.slot_accs_placeholder = tf.placeholder(shape=[None], dtype=tf.float32)
            self.intent_accs_op = tf.reduce_mean(self.intent_accs_placeholder)
            self.slot_accs_op = tf.reduce_mean(self.slot_accs_placeholder)
            tf.summary.scalar("intent_acc", self.intent_accs_op)
            tf.summary.scalar("slot_acc", self.slot_accs_op)

    def step(self, sess, trarin_batch):
        """ perform each batch"""

        unziped = list(zip(*trarin_batch))

        output_feeds = [self.train_op, self.loss, self.decoder_prediction,
                        self.intent, self.mask]
        feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
                     self.encoder_inputs_actual_length: unziped[1],
                     self.decoder_targets: unziped[2],
                     self.intent_targets: unziped[3]}
        results = sess.run(output_feeds, feed_dict=feed_dict)
        return results

    def get_score(self, sess, intent_accs, slot_accs, merged_summary):
        results = sess.run([self.intent_accs_op, self.slot_accs_op, merged_summary],
                           feed_dict={
                               self.intent_accs_placeholder: intent_accs,
                               self.slot_accs_placeholder: slot_accs
                           })
        return results
