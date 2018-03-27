# 生成 数据集
from data_utils import k_fold
import numpy as np

with open("/Users/huangpeisong/Desktop/task-slu-tencent.dingdang/rnn/test_data.txt") as fp:
    raw_data = [v.split(' ') for v in fp.readlines()]

sentences = [v[1:v.index("EOS")] for v in raw_data]
slot_sentences = [v[v.index("EOS") + 2:-1] for v in raw_data]
labels = [v[-1].replace('\n', '') for v in raw_data]
train_X, train_slot_sentences, train_Y, test_X, test_slot_sentences, test_Y = k_fold(10, X=sentences, Y=labels,
                                                                                     slot_sentences=slot_sentences)

train_input = train_X[0]
train_slot = train_slot_sentences[0]
train_intent = train_Y[0]

test_input = test_X[0]
test_slot = test_slot_sentences[0]
test_intent = test_Y[0]

np.save("train_input", train_input)
np.save("train_slot", train_slot)
np.save("train_intent", train_intent)

np.save("test_input", test_input)
np.save("test_slot", test_slot)
np.save("test_intent", test_intent)
