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

for idx in range(len(joint_result)):
    if joint_result[idx] == result_true[idx]:
        correct += 1
    else:
        mistake += 1

print(correct/(correct+mistake))