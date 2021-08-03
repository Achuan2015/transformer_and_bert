import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query, key, value, scale=None, attn_mask=None):
        attention = torch.bmm(query, key.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.mask_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, value)
        return context, attention


class LayerNorm(nn.Module):

    def __init__(self, model_dim=512, eps=1e-9):
        super(LayerNorm, self).__init__()
        self.w_a = nn.Parameter(torch.ones(model_dim))
        self.w_b = nn.Parameter(torch.zeros(model_dim))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.w_a * (x - mean) / (std + self.eps) + self.w_b
        

class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.per_head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        
        self.final_linear = nn.Linear(model_dim, model_dim)
        # 实现 ADD&Norm
        self.layer_norm = LayerNorm(model_dim)
    
    def forward(self, key, value, query, attn_mask=None):
        residual = query
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)
        
        batch_size = query.size(0)
        query = query.view(batch_size * self.num_heads, -1, self.per_head_dim)
        key = key.view(batch_size * self.num_heads, -1, self.per_head_dim)
        value = value.view(batch_size * self.num_heads, -1, self.per_head_dim)
        scale = self.model_dim ** -0.5
        if attn_mask:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
        context, attention = self.scaled_dot_product_attention(query, key, value, scale, attn_mask)
        context = context.view(batch_size, -1, self.model_dim)
        output = self.final_linear(context)
        output = self.dropout(output)
        output =  self.layer_norm(residual + output)
        return output, attention


def padding_mask(input_q, input_k):
    q_len = input_q.size(1)
    mask = input_k.eq(0)
    mask = mask.unsqueeze(1).expand(-1, q_len, -1)
    return mask


def sequence_mask(input):
    batch_size = input.size(0)
    seq_len = input.size(1)
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class PositionalEncoding(nn.Module):

    def __init__(self, model_dim=512, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        self.max_seq_len = max_seq_len
        self.position_encode = torch.Tensor([[pos / np.power(10000, 2 * (j // 2) / model_dim) for j in range(model_dim)] for pos in range(max_seq_len)])
        self.position_encode[:, 0::2] = torch.sin(self.position_encode[:, 0::2])
        self.position_encode[:, 1::2] = torch.cos(self.position_encode[:, 1::2])
        pad_row = torch.zeros(1, model_dim)
        # pad_row 放在前面，把pos=0的位置留给 pad_row， 方便padding_index 填充为0
        self.position_encode = torch.cat((pad_row, self.position_encode), 0)
        self.position_embedding = nn.Embedding(max_seq_len + 1, model_dim)
        self.position_embedding.weight = nn.Parameter(self.position_encode, requires_grad=False)
    
    def forward(self, input_len):
        # 确定pos embedding 的类型
        inputs = [list(range(1, seq_len + 1)) + [0] * (self.max_seq_len - seq_len) for seq_len in input_len]
        tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        inputs = tensor(inputs)
        return self.position_embedding(inputs)
