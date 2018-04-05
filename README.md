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
