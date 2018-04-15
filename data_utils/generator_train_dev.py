import numpy as np
from sklearn.model_selection import KFold

# label_num_10 = {
#     'OTHERS': 659.8,
#     'music.next': 13.2,
#     'music.pause': 30.0,
#     'music.play': 642.5,
#     'music.prev': 0.5,
#     'navigation.cancel_navigation': 83.5,
#     'navigation.navigation': 396.1,
#     'navigation.open': 24.5,
#     'navigation.start_navigation': 3.3,
#     'phone_call.cancel': 2.2,
#     'phone_call.make_a_phone_call': 279.6,
# }
#
# label_num_all = {
#     'OTHERS': 6598,
#     'music.next': 132,
#     'music.pause': 300,
#     'music.play': 6425,
#     'music.prev': 5,
#     'navigation.cancel_navigation': 835,
#     'navigation.navigation': 3961,
#     'navigation.open': 245,
#     'navigation.start_navigation': 33,
#     'phone_call.cancel': 22,
#     'phone_call.make_a_phone_call': 2796,
# }
#

kf = KFold(10, shuffle=False, random_state=223)
X = np.load("../10_fold_corpus/train_dev_X.npy")
Y = np.load("../10_fold_corpus/train_dev_Y.npy")

train_index = []
test_index = []
test_X = []
test_Y = []
train_X = []
train_Y = []
for train_idx, test_idx in kf.split(X):
    train_index.append(train_idx)
    test_index.append(test_idx)

for i in range(len(train_index)):
    train_X.append([X[v] for v in train_index[i]])
    train_Y.append([Y[v] for v in train_index[i]])

for i in range(len(test_index)):
    test_X.append([X[v] for v in test_index[i]])
    test_Y.append([Y[v] for v in test_index[i]])


name = '_data_'

for i in range(10):
    np.save("../10_fold_corpus/train_X" + name + str(i), train_X[i])
    np.save("../10_fold_corpus/train_Y" + name + str(i), train_Y[i])
    np.save("../10_fold_corpus/test_X" + name + str(i), test_X[i])
    np.save("../10_fold_corpus/test_Y" + name + str(i), test_Y[i])
print("generator_finish")
