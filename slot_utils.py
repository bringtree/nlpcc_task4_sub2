import jieba
import re

# 假的slot_word_set。里面存在||这样的词
tmp_slot_word_set = set()
# 真的slot_word_set
slot_word_set = set()
target_words_set = set()

row_data = []
with open("corpus.train.txt") as fp:
    tmp = fp.readlines()
    for v in tmp:
        if v == '\n':
            continue
        row_data.append(v.split('\t'))

not_cut_sentences_list = [v[1] for v in row_data]
label_list = [v[2] for v in row_data]
not_cut_slot_sentences_list = [v[3] for v in row_data]
cut_slot_sentences_list = []

target_words_pattern = "(?<=<).+?(?=\>)"
for sentences in not_cut_slot_sentences_list:
    target_words_set |= set(re.findall(target_words_pattern, sentences))

for slot_sentence in not_cut_slot_sentences_list:
    reg_result = None
    for target_word in target_words_set:
        pattern = "<" + target_word + ">.*?</" + target_word + ">"
        reg_result = re.findall(pattern, slot_sentence)
        tmp_slot_word_set |= set(reg_result)

# 这里jieba分词有bug https://github.com/fxsjy/jieba/issues/300
# for O_O in slot_word_set:
#     jieba.add_word(O_O)
# tmp_slot_sentence = []
# for sentence in not_cut_slot_sentences_list:
#     tmp_slot_sentence.append(jieba.lcut(sentence))

# 深拷贝一次，里面存在||一起的同义词
slot_word_set |= tmp_slot_word_set

for key_word in slot_word_set:
    for idx, not_cut_slot_sentence in enumerate(not_cut_slot_sentences_list):
        if key_word in not_cut_slot_sentence:
            type_pattern = "(?<=<)[a-z\_]+(?=\>)"
            type_key_word = re.search(type_pattern, key_word).group(0)
            content_word_pattern = "(<" + type_key_word + ">).*?(</" + type_key_word + ">)"
            content_word = re.search(content_word_pattern, key_word).group(0)
            not_cut_slot_sentences_list[idx] = not_cut_slot_sentences_list[idx].replace(content_word, type_key_word)

# 代码位置不可以调换，存在|| 的词
for word in tmp_slot_word_set:
    if '||' in word:
        slot_word_set.remove(word)
        pre_sign = re.search("<[a-z]+>", word).group(0)
        end_sign = re.search("</[a-z]+>", word).group(0)
        words = word.split("||")
        words[0] += end_sign
        words[1] = pre_sign + words[1]
        slot_word_set |= set(words)

tmp_sentences = []
for O_O in target_words_set:
    jieba.add_word(O_O)
for sentence in not_cut_slot_sentences_list:
    tmp_sentences.append(jieba.lcut(sentence))

for sentence in tmp_sentences:
    tmp = []
    for word in sentence:
        if word in target_words_set:
            tmp.append(word)
        else:
            tmp.append("O")
    cut_slot_sentences_list.append(" ".join(tmp))

# 我发现的问题 slot中的数据集有些类别是没有的。然后还有个问题。切是要根据slot切还是数据集切？ 我现在根据数据集切了一份。
