import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.w_o(context)
        return output, attn_weights

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_residual=True, use_layer_norm=True):
        super(TransformerEncoderLayer, self).__init__()
        
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        
        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with conditional residual and layer norm
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        
        if self.use_residual:
            x = x + self.dropout(attn_output)
        else:
            x = self.dropout(attn_output)
            
        if self.use_layer_norm:
            x = self.norm1(x)
        
        # FFN with conditional residual and layer norm
        ffn_output = self.ffn(x)
        
        if self.use_residual:
            x = x + self.dropout(ffn_output)
        else:
            x = self.dropout(ffn_output)
            
        if self.use_layer_norm:
            x = self.norm2(x)
        
        return x, attn_weights

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_residual=True, use_layer_norm=True):
        super(TransformerDecoderLayer, self).__init__()
        
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # 确保这些属性被正确初始化
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        
        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        else:
            # 确保即使不使用layer norm，这些属性也存在
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        
        if self.use_residual:
            x = x + self.dropout1(self_attn_output)
        else:
            x = self.dropout1(self_attn_output)
            
        x = self.norm1(x)
        
        # Cross-attention - 添加空值检查
        cross_attn_output, cross_attn_weights = None, None
        if encoder_output is not None:
            cross_attn_output, cross_attn_weights = self.cross_attn(x, encoder_output, encoder_output, src_mask)
            if self.use_residual:
                x = x + self.dropout2(cross_attn_output)
            else:
                x = self.dropout2(cross_attn_output)
                
            x = self.norm2(x)
        
        # FFN
        ffn_output = self.ffn(x)
        
        if self.use_residual:
            x = x + self.dropout3(ffn_output)
        else:
            x = self.dropout3(ffn_output)
            
        x = self.norm3(x)
        
        return x, self_attn_weights, cross_attn_weights

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, 
                 max_seq_length, dropout=0.1, use_positional_encoding=True):
        super(TransformerEncoder, self).__init__()
        
        self.use_positional_encoding = use_positional_encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
            
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = self.token_embedding(x)
        
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
            
        x = self.dropout(x)
        
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)
            
        return x, all_attn_weights

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_seq_length, dropout=0.1, use_positional_encoding=True,
                 use_residual=True, use_layer_norm=True):
        super(TransformerDecoder, self).__init__()
        
        self.use_positional_encoding = use_positional_encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
            
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, use_residual, use_layer_norm)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.token_embedding(x)
        
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
            
        x = self.dropout(x)
        
        all_self_attn_weights = []
        all_cross_attn_weights = []
        
        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(x, encoder_output, src_mask, tgt_mask)
            all_self_attn_weights.append(self_attn_weights)
            all_cross_attn_weights.append(cross_attn_weights)
            
        return x, all_self_attn_weights, all_cross_attn_weights

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, 
                 d_ff, num_layers, max_seq_length, dropout=0.1,
                 use_positional_encoding=True, use_residual=True, use_layer_norm=True):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length, 
            dropout, use_positional_encoding
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length, 
            dropout, use_positional_encoding, use_residual, use_layer_norm
        )
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output, _ = self.encoder(src, src_mask)
        decoder_output, _, _ = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.output_projection(decoder_output)
        return output

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_seq_length, dropout=0.1, use_positional_encoding=True,
                 use_residual=True, use_layer_norm=True):
        super(TransformerLM, self).__init__()
        self.decoder = TransformerDecoder(
            vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length,
            dropout, use_positional_encoding, use_residual, use_layer_norm
        )
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        decoder_output, _, _ = self.decoder(x, None, None, mask)
        output = self.output_projection(decoder_output)
        return output
