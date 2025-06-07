import os
import random
import numpy as np
import wave
import torch.nn.utils.rnn as rnn
DEFAULT_CONFIG_FILENAME = 'asrt_config.json'
Max_length = 44
import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
import json
from vocab import ch2id, id2ch, py2id, id2py, sy2id, id2sy, sy2ids
from pytosy import py2sy, sy2py
import librosa


_config_dict = None

def load_config_file(filename: str) -> dict:
    '''
    加载json配置文件
    参数：\\
        filename: 文件名
    返回：\\
        配置信息字典
    '''
    global _config_dict

    with open(filename, 'r', encoding="utf-8") as file_pointer:
        _config_dict = json.load(file_pointer)
    return _config_dict

def normalize(yt):
    yt_max = np.max(yt)
    yt_min = np.min(yt)
    a = 1.0 / (yt_max - yt_min)
    b = -(yt_max + yt_min) / (2 * (yt_max - yt_min))

    yt = yt * a + b
    return yt

def extract_feature(input_file, feature='mfcc', dim=20, cmvn=True, delta=True, delta_delta=True,
                    window_size=25, stride=10, save_feature=None):
    y, sr = librosa.load(input_file, sr=16000)#Aishell默认为16000hz
    yt, _ = librosa.effects.trim(y, top_db=20)
    yt = normalize(yt)
    ws = int(sr * 0.001 * window_size)
    st = int(sr * 0.001 * stride)
    if feature == 'mfcc':
        feat = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=dim,
                                    n_fft=ws, hop_length=st)
        feat[0] = librosa.feature.rms(y=yt, hop_length=st, frame_length=ws)

    else:
        raise ValueError('Unsupported Acoustic Feature: ' + feature)

    feat = [feat]

    if delta:
        feat.append(librosa.feature.delta(feat[0]))

    if delta_delta:
        feat.append(librosa.feature.delta(feat[0], order=2))

    feat = np.concatenate(feat, axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]
    if save_feature is not None:
        tmp = np.swapaxes(feat, 0, 1).astype('float32')
        np.save(save_feature, tmp)
        return len(tmp)
    else:
        return np.swapaxes(feat, 0, 1).astype('float32')

# Source: https://www.kaggle.com/davids1992/specaugment-quick-implementation
def spec_augment(spec: np.ndarray,
                 num_mask=2,
                 freq_masking=0.15,
                 time_masking=0.20,
                 value=0):
    spec = spec.copy()
    num_mask = random.randint(1, num_mask)
    for i in range(num_mask):
        all_freqs_num, all_frames_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[f0:f0 + num_freqs_to_mask, :] = value

        time_percentage = random.uniform(0.0, time_masking)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[:, t0:t0 + num_frames_to_mask] = value
    return spec

def pad_collate(batch):
    # 提取特征和标签，避免多余拷贝
    features = [torch.as_tensor(item[0]) for item in batch]  # List of (T, 60) tensors
    pinyin_labels = [torch.as_tensor(item[1]) for item in batch]  # List of pinyin labels
    sy_labels = [torch.as_tensor(item[2]) for item in batch] # List of syllable labels
    char_labels = [torch.as_tensor(item[3]) for item in batch]  # List of character labels

    features_length = torch.as_tensor([len(f) for f in features])

    # 填充特征
    features_padded = rnn.pad_sequence(features, batch_first=True, padding_value=0)  # Shape: (B, T, 60)

    # 填充标签，使用 .clone().detach() 避免不必要的拷贝

    pinyin_labels_padded = rnn.pad_sequence(
        [label.clone().detach() for label in pinyin_labels], batch_first=True, padding_value=-1
    )
    sy_labels_padded = rnn.pad_sequence(
        [label.clone().detach() for label in sy_labels], batch_first=True, padding_value=-1
    )

    char_labels_padded = rnn.pad_sequence(
        [label.clone().detach() for label in char_labels], batch_first=True, padding_value=0
    )

    # 计算长度
    pinyin_lengths = torch.as_tensor([len(label) for label in pinyin_labels])
    sy_lengths = torch.as_tensor([len(label) for label in sy_labels])
    char_lengths = torch.as_tensor([len(label) for label in char_labels])

    # 使用广播创建 mask，避免不必要的中间张量
    mask = (torch.arange(features_padded.size(1), device=features_padded.device)
            .expand(len(features_length), features_padded.size(1)) < features_length.unsqueeze(1))

    result = {
        'features': features_padded,               # Shape: (B, T, 60)
        'pinyin_labels': pinyin_labels_padded,     # Shape: (B, max_pinyin_length)
        'sy_labels': sy_labels_padded,             # Shape: (B, max_sy_length)
        'char_labels': char_labels_padded,         # Shape: (B, max_char_length)
        'pinyin_lengths': pinyin_lengths,          # Shape: (B,)
        'sy_lengths': sy_lengths,                  # Shape: (B,)
        'char_lengths': char_lengths,              # Shape: (B,)
        'mask': mask
    }
    return result


# 目标加载语音/拼音/文本数据集
class Dataloader(Dataset):
    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type
        self.datalist = list()
        self.wav_dict = dict()
        self.pinyin_dict = dict()
        self.wav_data = dict()
        self.char_dict = dict()
        self.sy_dict = dict()
        self._load_data()

    def _load_data(self):
        config = load_config_file(DEFAULT_CONFIG_FILENAME)#加载JSON
        for index in range(len(config['dataset'][self.dataset_type])):
            filename_datalist = config['dataset'][self.dataset_type][index]['data_list']
            filename_datapath = config['dataset'][self.dataset_type][index]['data_path']#./data

            #提取语音目录，并读取
            with open(filename_datalist, 'r', encoding='utf-8') as file_pointer:
                lines = file_pointer.read().split('\n')
                for line in lines:
                    if len(line) == 0:
                        continue
                    tokens = line.split(' ')
                    self.datalist.append(tokens[0])
                    self.wav_data[tokens[0]] = dict()
                    self.wav_dict[tokens[0]] = os.path.join(filename_datapath, tokens[1])

            filename_pinyinlist = config['dataset'][self.dataset_type][index]['pinyin_list']
            with open(filename_pinyinlist, 'r', encoding='utf-8') as file_pointer:
                lines = file_pointer.read().split('\n')
                for line in lines:
                    if len(line) == 0:
                        continue
                    tokens = line.split(' ')
                    py = tokens[1:]
                    pytemp = []
                    sytemp = []
                    for i in py:
                        pytemp.append(py2id[i])
                        for y in sy2ids(i):
                            sytemp.append(y)
                    self.pinyin_dict[tokens[0]] = pytemp
                    self.sy_dict[tokens[0]] = sytemp

            filename_charlist = config['dataset'][self.dataset_type][index]['char_list']
            #提取文本目录，将文本内容与其对应，需要将文本内容与vocab对应
            with open(filename_charlist, 'r', encoding='utf-8') as file_pointer:
                for line in file_pointer:
                    line = line.rstrip('\n')
                    if not line.strip():
                        continue
                    tokens = line.split(' ')
                    text = tokens[1:]
                    chtemp = [4329]
                    for i in text:
                        chtemp.append(ch2id[i])
                    chtemp.append(4330)
                    self.char_dict[tokens[0]] = chtemp


    def __getitem__(self, i):

        mark = self.datalist[i]
        wav_file = self.wav_dict[mark]
        pinyin_label = self.pinyin_dict[mark]
        sy_label = self.sy_dict[mark]
        char_label = self.char_dict[mark]
        feature = extract_feature(input_file=wav_file, feature='mfcc',dim = 20,cmvn = True)
        feature = (feature - feature.mean()) / feature.std()
        if self.dataset_type == "train":
            feature = spec_augment(feature)

        return feature, pinyin_label, sy_label, char_label

    def __len__(self):
        return len(self.datalist)

if __name__ == "__main__":
    train_dataset = Dataloader('train')
    print("Datalist length:", len(train_dataset.datalist))  # 确保数据加载正常
    print("First data item:", train_dataset[0])  # 检查第一个样本是否正常
    dev_dataset = Dataloader('dev')
    print("Datalist length:", len(dev_dataset.datalist))  # 确保数据加载正常
    print("First data item:", dev_dataset[0])  # 检查第一个样本是否正常

