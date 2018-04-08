import os
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt

import os

for k_fold_index in range(10):
    name = 'result_list_' + str(k_fold_index)
    with open("./result/result" + str(k_fold_index) + ".txt", "r") as fp:
        locals()[name] = fp.readlines()

with open("./result/resulttrue.txt", "r") as fp:
    result_true = fp.readlines()

with open("./result/sentence.txt", "r") as fp:
    sentence = fp.readlines()

joint_result = []
for result_predict_idx in range(len(locals()["result_list_0"])):
    label_list = {
        "OTHERS\n": 0,
        "music.next\n": 0,
        "music.pause\n": 0,
        "music.play\n": 0,
        "music.prev\n": 0,
        "navigation.cancel_navigation\n": 0,
        "navigation.navigation\n": 0,
        "navigation.open\n": 0,
        "navigation.start_navigation\n": 0,
        "phone_call.cancel\n": 0,
        "phone_call.make_a_phone_call\n": 0
    }
    for k_fold_index in range(10):
        name = 'result_list_' + str(k_fold_index)
        label_list[str(locals()[name][result_predict_idx])] += 1
    joint_result.append(sorted(label_list, key=lambda x: label_list[x])[-1])

# with open('./result/joint.txt', "w") as fp:
#     fp.writelines(joint_result)

correct = 0
mistake = 0
truth_table = []
for idx in range(len(joint_result)):
    if joint_result[idx] == result_true[idx]:
        truth_table.append("TRUE")
        correct += 1
    else:
        truth_table.append("FALSE")
        mistake += 1

# print(correct/(correct+mistake))

columns_order = ["sentence","correct","result_true","joint_result"]
result_dict = {}
result_dict["sentence"] = sentence
result_dict["correct"] = truth_table
result_dict["result_true"] = result_true
result_dict["joint_result"] = joint_result
for k_fold_index in range(10):
    name = 'result_list_' + str(k_fold_index)
    columns_order.append(name)
    result_dict[name] = locals()[name]


f1 = DataFrame(result_dict,columns=columns_order)

f1.to_csv("./test_result.csv",encoding="utf-8")