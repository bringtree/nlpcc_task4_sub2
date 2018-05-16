```
.
├── 10_fold_corpus（K-fold数据集）
│   ├── test_X_data_0.npy
│   ├── test_X_data_1.npy
│   ├── test_X_data_2.npy
│   ├── test_X_data_3.npy
│   ├── test_X_data_4.npy
│   ├── test_X_data_5.npy
│   ├── test_X_data_6.npy
│   ├── test_X_data_7.npy
│   ├── test_X_data_8.npy
│   ├── test_X_data_9.npy
│   ├── test_Y_data_0.npy
│   ├── test_Y_data_1.npy
│   ├── test_Y_data_2.npy
│   ├── test_Y_data_3.npy
│   ├── test_Y_data_4.npy
│   ├── test_Y_data_5.npy
│   ├── test_Y_data_6.npy
│   ├── test_Y_data_7.npy
│   ├── test_Y_data_8.npy
│   ├── test_Y_data_9.npy
│   ├── train_X_data_0.npy
│   ├── train_X_data_1.npy
│   ├── train_X_data_2.npy
│   ├── train_X_data_3.npy
│   ├── train_X_data_4.npy
│   ├── train_X_data_5.npy
│   ├── train_X_data_6.npy
│   ├── train_X_data_7.npy
│   ├── train_X_data_8.npy
│   ├── train_X_data_9.npy
│   ├── train_Y_data_0.npy
│   ├── train_Y_data_1.npy
│   ├── train_Y_data_2.npy
│   ├── train_Y_data_3.npy
│   ├── train_Y_data_4.npy
│   ├── train_Y_data_5.npy
│   ├── train_Y_data_6.npy
│   ├── train_Y_data_7.npy
│   ├── train_Y_data_8.npy
│   └── train_Y_data_9.npy
├── README.md
├── data (会用到的数据)
│   ├── corpus.train.txt (原始的数据)
│   ├── labels.txt （label的种类，不要去改动到。会影响编码）
│   ├── new_paragraph_sentences.npy 
│   ├── pad_new_paragraph_labels.npy
│   ├── slot-dictionaries (原始的数据)
│   │   ├── age.txt
│   │   ├── custom_destination.txt
│   │   ├── emotion.txt
│   │   ├── instrument.txt
│   │   ├── language.txt
│   │   ├── scene.txt
│   │   ├── singer.txt
│   │   ├── song.txt
│   │   ├── style.txt
│   │   ├── theme.txt
│   │   └── toplist.txt
│   ├── vec_dict.pkl (词向量的编码字典)
│   ├── word_dict.pkl (词-数字的字典)
│   ├── word_dict_reverse.pkl (数字-词的字典)
│   └── words_list.txt(slot-dictionaries中的词汇总)
├── data_utils
│   ├── context_slot.py (生成new_paragraph_sentences.npy/pad_new_paragraph_labels.npy的代码)
│   ├── generator_dict.py  (生成 词-字 字典)
│   ├── generator_vec_dict.py (生成词向量的字典)
│   ├── generator_words.py （ 把slot-dictionaries中的词汇总成words_list.txt的代码，没事别运行）
│   └── k-fold.py （k折代码，没事别去运行）
├── ltp_data_v3.4.0 (分词的模块)
│   ├── cws.model
│   ├── md5.txt
│   ├── version
│   └── word_split.py
├── main_no_embedding.py (训练的代码)
├── model_graph (我之前测试图用的。无视。我自用)
└── model_hrnn.py （h-rnn的模型代码）
```

# 2018-nlpcc task4  子任务2 第四名 f1分数 0.929方案

模型结构 ： 1个embedding + 2个双向的lstm 训练句子意图 + 1个单向lstm处理输出经过上文影响后的句子意图。

训练使用网上的 fasttext 预先训练的词向量。 

模型参数全部随缘。就试了个学习率- - 0.0001 -> 0.001

融合模型：训练 10个模型  联合投票

# 训练注意的地方：

1 分开训练 先把 下面的句子的lstm + 和 上面的上下文lstm 一起训练。 之后loss稳定下来后 把下面的lstm 固定住 只训练上层lstm。 最后把所有lstm 固定住 再训练一遍词向量。

2 学习率调大点。 大力出奇迹

# 也许能提升分数的地方(没做 写完模型 就去看其他东西了 O(∩_∩)O！) 

1 模型没有利用到任何实体信息(官方是有提供的。我刚开始以为是给slot任务的- -)。 后来想了想 可以在词向量上 下点功夫。比如 再开个 几个维度 来标注下词向量的实体信息。

2 可以把adam 换成sgd 感觉 分数会更高

# 做过一些事情，结果分数反而下降

1 交上去的模型 后2份在模型的基础上做了一些正则匹配。结果 分数反而低了。 可能误杀了一些。

2 对数据做了一些增强，比如 替换 等等 分数下降

3 loss加了个L2正则化，下降

4 学习率调得低， 分数下降

