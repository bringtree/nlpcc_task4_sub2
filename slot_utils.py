import jieba
import re

# 假的slot_word。里面存在||这样的词
tmp_slot_word = set()
# 真的slot_word
slot_word = set()

row_data = []
with open("corpus.train.txt") as fp:
    tmp = fp.readlines()
    for v in tmp:
        if v == '\n':
            continue
        row_data.append(v.split('\t'))

not_cut_sentences = [v[1] for v in row_data]
label = [v[2] for v in row_data]
slot_sentences = [v[3] for v in row_data]


def judge_target_word(target_sentence):
    """

    :param target_sentence:
    :return:
    """
    global tmp_slot_word
    target_words = ['toplist',
                    'theme',
                    'style',
                    'song',
                    'singer',
                    'scene',
                    'language',
                    'instrument',
                    'emotion',
                    'custom_destination',
                    'age']
    reg_result = None
    for target_word in target_words:
        pattern = "<" + target_word + ">.*?</" + target_word + ">"
        reg_result = re.findall(pattern, target_sentence)
        tmp_slot_word |= set(reg_result)


for slot_sentence in slot_sentences:
    judge_target_word(slot_sentence)

# 存在|| 的词
slot_word |= tmp_slot_word
for word in tmp_slot_word:
    if '||' in word:
        slot_word.remove(word)
        pre_sign = re.search("<[a-z]+>", word).group(0)
        end_sign = re.search("</[a-z]+>", word).group(0)
        words = word.split("||")
        words[0] += end_sign
        words[1] = pre_sign + words[1]
        slot_word |= set(words)
