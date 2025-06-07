import torch
import torch.nn as nn
import torch.nn.functional as F

class PinyinEmbeddingCNN(nn.Module):
    def __init__(self, vocab_size=32, char_emb_dim=16, cnn_out_dim=128, kernel_size=3, padding=1):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, char_emb_dim, padding_idx=0)
        self.conv = nn.Conv1d(in_channels=char_emb_dim, out_channels=cnn_out_dim, kernel_size=kernel_size, padding=padding)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, pinyin_seq):  # pinyin_seq: (B, T, 8)
        B, T, L = pinyin_seq.shape
        x = self.char_emb(pinyin_seq)  # (B, T, 8, char_emb_dim)
        x = x.view(B * T, L, -1).transpose(1, 2)  # (B*T, char_emb_dim, 8)
        x = self.conv(x)  # (B*T, cnn_out_dim, 8)
        x = F.relu(x)
        x = self.pool(x).squeeze(-1)  # (B*T, cnn_out_dim)
        x = x.view(B, T, -1)  # (B, T, cnn_out_dim)
        return x  # yemb_i




def pad_pinyin(pinyin_strs, pad_len=8, pad_char='-'):
    # pinyin_strs: list of strings, e.g. ['zhuang', 'guo']
    return [list(p[:pad_len].ljust(pad_len, pad_char)) for p in pinyin_strs]


