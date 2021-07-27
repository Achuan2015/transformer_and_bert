"""
基于Pytorch实现Transformer
"""

import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention Mechanism"""
    
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(-2, -1)) # B * L * L
        if scale:
            attention = attention * scale
        if attn_mask:
            # 给attn_mask 中bool 值为True的位置加上负无穷的数
            # attention -> (B * num_heads) * L * L 
            attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax 权重
        attention = self.softmax(attention)
        # 增加一层dropout
        attention = self.dropout(attention)
        # 计算output vector -> context
        # (B * L * L) * (B * L * D) -> B * L * D
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.per_head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.linear_q = nn.Linear(model_dim, self.per_head_dim * num_heads)
        self.linear_k = nn.Linear(model_dim, self.per_head_dim * num_heads)
        self.linear_v = nn.Linear(model_dim, self.per_head_dim * num_heads)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        
    def forward(self, key, value, query, attn_mask=None):
        # 准备残差网络的输入
        residual = query
        # 分别增加query, key, value 的学习参数: linear project
        query =  self.linear_q(query)
        key =  self.linear_k(key)
        value =  self.linear_v(value)
        # splited by heads
        batch_size = query.size(0)
        query = query.view(batch_size * self.num_heads, -1, self.per_head_dim)
        key = key.view(batch_size * self.num_heads, -1, self.per_head_dim)
        value = value.view(batch_size * self.num_heads, -1, self.per_head_dim)
        # 准备scale dot product attention 的输入参数
        scale = self.model_dim ** -0.5
        if attn_mask:
            # 因为 query，key，value根据 num_heads 进行的split，因此 attn_mask 需要扩展到相同的维度
            # attn_mask -> B * L * L after repeat -> (num_heads * B) * L * L
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
        context, attention = self.scaled_dot_product_attention(query, key, value, scale, attn_mask)
        
        # 从新将heads concate 起来
        context = context.view(batch_size, -1, self.num_heads * self.per_head_dim)
        # context 在output 之前还要进行一次线性映射
        output = self.linear_final(context)
        # 最后还需要进行dropout
        output = self.dropout(context)
        # 每一层输出之前都要进行 ADD & NORM
        output = self.layer_norm(residual + output)
        return output, attention


def padding_mask(seq_q, seq_k):
    """
    （1）选择q，k是因为 k，v的来源总是一致的，因此seq_k 与 seq_v 任意选择就好了
    （2）在encode-decoder-attention 中seq_query 来源不是一致的
    """
    q_len = seq_q.size(1)
    pad_mask = seq_k.eq(0) # bool matrix
    # （1）说明了为啥需要seq_q 与 seq_k
    # pad_mask 用在 softmax 之前，在dot-product之后 attention维度为 B * q_len * k_len，所以pad_mask 应该与这个一致
    pad_mask = pad_mask.unsqueeze(1).expand(-1, q_len, -1) # B * q_len * k_len  
    return pad_mask

def sequence_mask(seq):
    pass




if __name__ == "__main__":
    # # test1
    # scaled_self_attention = ScaledDotProductAttention(0.1)
    # q, k, v = [torch.randn(10, 20, 200) for _ in range(3)]
    # scale = 1 / torch.sqrt(torch.tensor(200, dtype=torch.float32))
    # output, attention = scaled_self_attention(q, k, v, scale)
    # print(output.shape)
    # # test2
    # query = torch.randn(10, 128, 512)
    # multi_head_attention = MultiHeadAttention(dropout=0.1)
    # output, attention = multi_head_attention(query, query, query)
    # print(output.shape, attention.shape)