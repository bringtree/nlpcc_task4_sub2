我切割数据集的时候存在一些问题

1：存在一些slot-dictionaries:song中不存在的词 如 什话

2：存在一些不存在的slot-dictionaries类别(扫了一眼挺多的) 如 destination

3：英文单词(歌手名等等一些专有名词中存在空格，我觉得会影响分词，我全部替换成的#，如 m c替换成m#c)

4: slot-dictionaries:中的词要怎么处理合适。

5: test_data.txt 没按slot-dictionaries的值去切，直接按数据集去切了


阅读代码遇到的问题。

求助中。。。
https://github.com/applenob/RNN-for-Joint-NLU/issues/6

记录一个bug
还没有做