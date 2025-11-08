import torch
from torch.utils.data import Dataset
import numpy as np

class TextDataset(Dataset):
    def __init__(self, file_path, seq_length):
        self.seq_length = seq_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # 修复：索引从0开始，添加padding字符
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # 添加padding字符
        self.char_to_idx['<pad>'] = 0
        self.idx_to_char[0] = '<pad>'
        self.vocab_size = len(self.chars) + 1  # 包括padding
        
        self.data = [self.char_to_idx[ch] for ch in text]
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        
        return torch.tensor(x), torch.tensor(y)

def create_masks(src, tgt, pad_idx=0):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len))).bool()
    tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0).unsqueeze(0)
    
    return src_mask, tgt_mask
