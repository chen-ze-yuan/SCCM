import os
from pytosy import py2sy, sy2py

# 定义文件路径
new_aishell_path = './data/data_aishell/transcript/new_aishell.txt'
pinyin_path = './data/data_aishell/transcript/pinyin.txt'

# 创建空的词典集合
text_dict = set()
pinyin_dict = set()
sy_dict = set()

# 读取文本词典
with open(new_aishell_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) > 1:
            text_dict.update(parts[1:])

# 读取拼音和声韵母词典
with open(pinyin_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) > 1:
            pinyin_list = parts[1:]
            pinyin_dict.update(pinyin_list)
            for pinyin in pinyin_list:
                if pinyin in py2sy:
                    sy_dict.update(py2sy[pinyin])

ch2id = {word: idx for idx, word in enumerate(sorted(text_dict), start=1)}
ch2id['<sos>'] = 4329
ch2id['<eos>'] = 4330
py2id = {'<pad>': 0, **ch2id}
id2ch = {idx: word for word, idx in ch2id.items()}
print(len(ch2id))#4330
#print(ch2id)

py2id = {word: idx for idx, word in enumerate(sorted(pinyin_dict), start=1)}
py2id = {'<blank>': 0, **py2id}
id2py = {idx: word for word, idx in py2id.items()}
print(len(py2id))#403
#print(py2id)

sy2id = {word: idx for idx, word in enumerate(sorted(sy_dict), start=1)}
sy2id = {'<blank>': 0, **sy2id}
id2sy = {idx: word for word, idx in sy2id.items()}
print(len(sy2id))#65
print(sy2id)

def sy2ids(pinyin):
    list = py2sy[pinyin]
    syllable_ids = [sy2id[syllable] for syllable in list]
    return syllable_ids
