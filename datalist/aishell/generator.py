import os
import string
import pypinyin
from pypinyin import lazy_pinyin,Style

def generate_classify(input_file, output_file, data_dir):
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            filename = line.split()[0] + '.wav'
            filepath = os.path.join(data_dir, filename[6:11], filename)
#           print(filepath)
            if os.path.exists(filepath):
                f_out.write(line)


# 生成 train_char.txt
generate_classify('/data/data_aishell/transcript/new_aishell.txt', 'train_char.txt', 'F:/first/data/train')

#生成 test_char.txt
generate_classify('/data/data_aishell/transcript/new_aishell.txt', 'test_char.txt', 'F:/first/data/test')

# 生成 dev_char.txt
generate_classify('/data/data_aishell/transcript/new_aishell.txt', 'dev_char.txt', 'F:/first/data/dev')

# 生成 train_char.txt
generate_classify('/data/data_aishell/transcript/pinyin.txt', 'train_pinyin.txt', 'F:/first/data/train')

#生成 test_char.txt
generate_classify('/data/data_aishell/transcript/pinyin.txt', 'test_pinyin.txt', 'F:/first/data/test')

# 生成 dev_char.txt
generate_classify('/data/data_aishell/transcript/pinyin.txt', 'dev_pinyin.txt', 'F:/first/data/dev')
