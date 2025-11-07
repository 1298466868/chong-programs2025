# src/data_utils.py
import torch
from torch.utils.data import Dataset
import numpy as np

class TextDataset(Dataset):
    def __init__(self, file_path, seq_length):
        self.seq_length = seq_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create character-level vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i+1 for i, ch in enumerate(self.chars)}  # 0 for padding
        self.idx_to_char = {i+1: ch for i, ch in enumerate(self.chars)}
        
        # Convert text to indices
        self.data = [self.char_to_idx[ch] for ch in text]
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        
        return torch.tensor(x), torch.tensor(y)

def create_masks(src, tgt, pad_idx=0):
    # Source padding mask
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    # Target padding mask and future mask
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len))).bool()
    tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0).unsqueeze(0)
    
    return src_mask, tgt_mask
