# model 是fasttext的模型。
import pickle
import numpy as np
import os

os.chdir("/Users/huangpeisong/Desktop/seq2seq_chatbot_new/hand")
with open("./data/word_dict.pkl","rb") as fp:
    word_dict = pickle.load(fp)


vec_dict = {}
for key,value in word_dict.items():
    vec_dict[value] = model.get_word_vector(key)