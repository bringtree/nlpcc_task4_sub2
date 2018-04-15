# 把所有的slot词汇总 用来生成那个words_list.txt，这样就可以一次性 导入到ltp的字典中去

import os

source_file = "../data/slot-dictionaries"
all_files_list = os.listdir(source_file)

all_words = []
for file_name in all_files_list:
    with open(source_file + "/" + file_name) as fp:
        for v in fp.readlines():
            all_words.append(v)

with open("../data/words_list.txt", "w") as fp:
    fp.writelines(all_words)
