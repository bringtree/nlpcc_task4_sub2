import jieba
import re

# 这里jieba分词有点问题 https://github.com/fxsjy/jieba/issues/300
# 如 v = "aaaa bbb" 加到字典里会变成 "aaaa"," ","bbb"


# tmp_slot_word_set。里面存在||这样的词 如  <song>神话||什话</song>
tmp_slot_word_set = set()
# slot_word_set. <song>神话</song> <song>什话</song>
slot_word_set = set()
# slot标签 如 "song"和"/song"
target_words_set = set()

# 每一行的数据
row_data = []
with open("corpus.train.txt") as fp:
    tmp = fp.readlines()
    for v in tmp:
        if v == '\n':
            continue
        row_data.append(v.split('\t'))

# 没有切分成词前的句子 把英文单词中的" "全部切成#, 单独的空格，也会被转成#， 如"m c 天佑" 就变成了 "m # c 天佑".
not_cut_sentences_list = [v[1].replace(' ', "#") for v in row_data]
# 把句子 切分成词
cut_sentences_list = []
# 意图标签
label_list = [v[2] for v in row_data]
# 没有切分成词前的槽句子 把英文单词中的" "全部切成#, 单独的空格，也会被转成#， 如"simply sunday" 就变成了 "simply#sunday".
not_cut_slot_sentences_list = [v[3].replace(' ', "#") for v in row_data]
# 把cut_sentences 切分成词的槽句子
cut_slot_sentences_list = []

# 匹配slot标签 如 "song"和"/song"
# 收集所有slot标签
target_words_pattern = "(?<=<).+?(?=\>)"
for sentences in not_cut_slot_sentences_list:
    target_words_set |= set(re.findall(target_words_pattern, sentences))

# 切出所有的 带有槽值的 值 如 "<song>什话||神话</song>"
for slot_sentence in not_cut_slot_sentences_list:
    reg_result = None
    for target_word in target_words_set:
        pattern = "<" + target_word + ">.*?</" + target_word + ">"
        reg_result = re.findall(pattern, slot_sentence)
        tmp_slot_word_set |= set(reg_result)

# 深拷贝一次，里面存在||一起的同义词
slot_word_set |= tmp_slot_word_set

# 把not_cut_slot_sentences_list槽句子中的 "播放不一样来一首<song>我们不一样</song>" 变成了 "播放不一样来一首song"
for key_word in slot_word_set:
    for idx, not_cut_slot_sentence in enumerate(not_cut_slot_sentences_list):
        if key_word in not_cut_slot_sentence:
            type_pattern = "(?<=<)[a-z\_.]+(?=\>)"
            type_key_word = re.search(type_pattern, key_word).group(0)
            content_word_pattern = "(<" + type_key_word + ">).*?(</" + type_key_word + ">)"
            content_word = re.search(content_word_pattern, key_word).group(0)
            not_cut_slot_sentences_list[idx] = not_cut_slot_sentences_list[idx].replace(content_word, type_key_word)

# 切出所有的 带有槽值的 值 如 "<song>什话</song>" "<song>神话</song>"
# 代码位置不可以调换上下代码，存在|| 的词
for word in tmp_slot_word_set:
    if '||' in word:
        slot_word_set.remove(word)
        pre_sign = re.search("<[a-z_.]+>", word).group(0)
        end_sign = re.search("</[a-z_.]+>", word).group(0)
        words = word.split("||")
        words[0] += end_sign
        words[1] = pre_sign + words[1]
        slot_word_set |= set(words)

# 把所有词添加到slot标签值 如song /song ,theme,/theme 添加到字典中，
tmp_sentences = []
for O_O in target_words_set:
    jieba.add_word(O_O)
# 把not_cut_slot_sentences_list槽句子中的   "播放不一样来一首song" 切成词语
for sentence in not_cut_slot_sentences_list:
    tmp_sentences.append(jieba.lcut(sentence))

# 把not_cut_slot_sentences_list槽句子中的 "播放 不一样 来 一首 song" 里面不是 slot值的 全部换成O 如 "O O O O song"然后放入cut_slot_sentences_list
# cut_slot_sentences_list 完成
for sentence in tmp_sentences:
    tmp = []
    for word in sentence:
        if word in target_words_set:
            tmp.append(word)
        else:
            tmp.append("O")
    # 去掉'\n'变成的O
    tmp.pop()
    cut_slot_sentences_list.append(" ".join(tmp))

# 准备切句子。先准备不能切割的词 如"播放 不一样 来 一首 我们不一样" 中的"我们不一样"，del_sign_slot_word_list存放的就是"我们不一样"
del_sign_slot_word_list = list()
for word in slot_word_set:
    pre_sign = re.search("<[a-z_.]+>", word).group(0)
    end_sign = re.search("</[a-z_.]+>", word).group(0)
    word = word.replace(pre_sign, '')
    word = word.replace(end_sign, '')
    del_sign_slot_word_list.append(word)

# 把"del_sign_slot_word_list存放的就是"的词加入字典，然后切句子为词语
for O_O in del_sign_slot_word_list:
    jieba.add_word(O_O)
for sentence in not_cut_sentences_list:
    cut_sentences_list.append(" ".join(jieba.lcut(sentence)))

# 组合成特定格式的数据集，"BOS" 会占用 一个O
end = []
for idx in range(len(cut_sentences_list)):
    end.append(
        "BOS " + cut_sentences_list[idx].replace('\n', '') + " EOS O " + cut_slot_sentences_list[idx].replace('\n',
                                                                                                              '') + " " +
        label_list[idx] + '\n')
# 写入
with open('test_data.txt', 'w') as fp:
    fp.writelines(end)
