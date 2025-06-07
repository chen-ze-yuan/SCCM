from tabnanny import check

import numpy as np
import torch
# torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.distributed.pipeline.sync.checkpoint import checkpoint
from torch.nn.functional import dropout
from tqdm import tqdm
import os
import torch.nn.functional as F
import torchaudio
import pypinyin
from pypinyin import lazy_pinyin, Style
from vocab import ch2id, id2ch, py2id, id2py, sy2ids, sy2id
from pytosy import py2sy, sy2py
from dataload import Dataloader, pad_collate
from model.encoder.conformer import ConformerEncoder
from model.decoder.transformer import TransformerDecoder
from ctcdecode import CTCBeamDecoder
import random

seed = 3407
random.seed(3407)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
else:
    device = "cpu"


def ctc_loss(logits, targets, input_lengths, target_lengths):
    # 计算CTC损失

    loss = F.ctc_loss(logits, targets, input_lengths, target_lengths)
    return loss

cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

def save_checkpoint(model, optimizer, epoch, checkpoint_path='checkpoint.pth'):
    """
    保存模型和优化器的状态
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(model, optimizer, checkpoint_path='checkpoint.pth'):
    """
    恢复训练从保存的检查点
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from epoch {epoch}")
        return model, optimizer, epoch
    else:
        print("No checkpoint found, starting from scratch.")
        return model, optimizer, 0  # 从头开始


Encodermodel = ConformerEncoder(d_model=512, d_ff=1024, cov_kernel_size=31, n_heads=8,
                                nblocks=6, pos_dropout=0.1, slf_attn_dropout=0.0, ffn_dropout=0.1,
                                residual_dropout=0.1, conv_dropout=0.0, macaron_style=True,
                                ffn_scale=0.5, conv_bias=True, positional_encoding=True, relative_positional=True,
                                conv_first=False, activation='glu')

CharDecoder = TransformerDecoder(vocab_size=len(ch2id), d_model=512, n_heads=4, d_ff=1024, memory_dim=512,
                                 activation=nn.ReLU())

optimizer = torch.optim.Adam(list(Encodermodel.parameters()) + list(CharDecoder.parameters()), lr=0.001)

Encodermodel = Encodermodel.to(device)

train_dataset = Dataloader('train')
dev_dataset = Dataloader('dev')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, collate_fn=pad_collate,
                                           pin_memory=True, shuffle=False, num_workers=1)

dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=4, collate_fn=pad_collate,
                                         pin_memory=True, shuffle=False, num_workers=1)


def dev(model, device, dev_loader, loss1,decoder):
    model.eval()
    decoder.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batchT in dev_loader:
            data_input = batchT['features'].to(device)
            pinyin_label = batchT['pinyin_labels'].to(device)
            sy_label = batchT['sy_labels'].to(device)
            char_label = batchT['char_labels'].to(device)
            pinyin_lengths = batchT['pinyin_lengths'].to(device)
            sy_lengths = batchT['sy_lengths'].to(device)
            char_lengths = batchT['char_lengths'].to(device)
            mask = batchT['mask'].to(device)

            py_flat = pinyin_label[pinyin_label != -1].view(-1)
            sy_flat = sy_label[sy_label != -1].view(-1)
            output1, mask, attn_weights, py_feat, sy_feat = model(data_input, mask)  # .float()

            py_log_probs = py_feat.permute(1, 0, 2).log_softmax(dim=-1).to(device)
            py_input_lengths = torch.as_tensor([mask[i].sum().item() for i in range(mask.size(0))]).to(device)
            loss_py = loss1(py_log_probs, py_flat, py_input_lengths, pinyin_lengths)

            sy_log_probs = sy_feat.permute(1, 0, 2).log_softmax(dim=-1).to(device)
            sy_input_lengths = torch.as_tensor([mask[i].sum().item() for i in range(mask.size(0))]).to(device)
            loss_sy = loss1(sy_log_probs, sy_flat, sy_input_lengths, sy_lengths)

            char_input = char_label[:, :-1]  # 输入给解码器，去掉最后一个 <eos>
            char_target = char_label[:, 1:]  # 作为交叉熵目标，去掉第一个 <sos>
            char_output = CharDecoder(targets=char_input, memory=output1, memory_mask=mask)
            loss_ch = cross_entropy(
                char_output.view(-1, char_output.size(-1)),
                char_target.reshape(-1)
            )

            dev_loss = 0.2 * loss_py + 0.2 * loss_sy + 0.6 * loss_ch

            total_loss += dev_loss.item()
    avg_loss = total_loss / len(dev_loader)
    return avg_loss


def train_net(model, device, train_loader, dev_loader, optimizer, loss1, checkpoint__path, decoder):
    best_loss = float('inf')
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    model.train()
    decoder.train()
    for epoch in range(start_epoch, 20):
        epoch_loss = 0.0
        count = 0
        for batch in train_loader:
            data_input = batch['features'].to(device)
            pinyin_label = batch['pinyin_labels'].to(device)
            sy_label = batch['sy_labels'].to(device)
            char_label = batch['char_labels'].to(device)
            pinyin_lengths = batch['pinyin_lengths'].to(device)
            sy_lengths = batch['sy_lengths'].to(device)
            char_lengths = batch['char_lengths'].to(device)
            mask = batch['mask'].to(device)

            # print(mask.shape)#[B,T]
            optimizer.zero_grad()
            py_flat = pinyin_label[pinyin_label != -1].view(-1)
            sy_flat = sy_label[sy_label != -1].view(-1)
            output1, mask, attn_weights, py_feat, sy_feat = model(data_input, mask)  # .float()

            py_log_probs = py_feat.permute(1, 0, 2).log_softmax(dim=-1)

            py_input_lengths = torch.as_tensor([mask[i].sum().item() for i in range(mask.size(0))]).to(device)
            loss_py = loss1(py_log_probs, py_flat, py_input_lengths, pinyin_lengths)

            sy_log_probs = sy_feat.permute(1, 0, 2).log_softmax(dim=-1)
            sy_input_lengths = torch.as_tensor([mask[i].sum().item() for i in range(mask.size(0))]).to(device)
            loss_sy = loss1(sy_log_probs, sy_flat, sy_input_lengths, sy_lengths)

            char_input = char_label[:, :-1]  # 输入给解码器，去掉最后一个 <eos>
            char_target = char_label[:, 1:]  # 作为交叉熵目标，去掉第一个 <sos>
            char_output = CharDecoder(targets=char_input, memory=output1, memory_mask=mask)
            loss_ch = cross_entropy(
                char_output.view(-1, char_output.size(-1)),
                char_target.reshape(-1)
            )

            train_loss = 0.2 * loss_py + 0.2 * loss_sy + 0.6 * loss_ch

            train_loss.backward()
            optimizer.step()

            if count % 100 == 0:
                print(f"Batch [{count + 1}/{30025}], Training Loss: {epoch_loss}")
            count = count + 1

            epoch_loss += train_loss.item()

        print(f"Epoch [{epoch + 1}/{20}], Training Loss: {epoch_loss}")

        dev_loss = dev(model, device, dev_loader, loss1, decoder)

        with open('loss.txt', 'a') as f:
            f.write(f"Epoch [{epoch + 1}/{20}], Training Loss: {epoch_loss}\n")
            f.write(f"Epoch {epoch + 1}, Validation Loss: {dev_loss}\n")
        print(f"Epoch [{epoch + 1}/{50}], Validation Loss: {dev_loss}")

        save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)
        # 保存阶段2模型
    print("Training finished.")


if __name__ == '__main__':
    checkpoint_path = 'checkpoint.pth'

    train_net(Encodermodel, device, train_loader, dev_loader, optimizer, ctc_loss, checkpoint_path,CharDecoder)

