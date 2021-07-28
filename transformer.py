"""
基于Pytorch实现Transformer
"""

import torch
from torch._C import uint8
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


class LayerNorm(nn.Module):
    """
    layer normalization 的实现
    """
    
    def __init__(self, model_dim=512, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 构造Layer normalization的参数
        self.a_2 = nn.Parameter(torch.ones(model_dim))
        self.b_2 = nn.Parameter(torch.zeros(model_dim))
        # 除以标准差时的平滑值
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return  self.a_2 * (x - mean) / (std + self.eps) + self.b_2

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
    """
    构造 序列mask，将序列mask应用在每一个序列上。

    param:
        seq: -> B * L
    """
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask.eq(0)


class PositionalEmbedding(nn.Module):
    
    def __init__(self, model_dim, max_seq_len):
        super(PositionalEmbedding, self).__init__()
        # 根据公式，构造PE矩阵 w_k * t (t 为 pos， w_k)
        position_encoding = [[pos / np.power(10000, 2 * (j // 2)/model_dim) for j in range(model_dim)] for pos in range(max_seq_len)]
        # 根据公式：奇数列使用余弦（cos），偶数列使用（sim）
        position_encoding[:, 0::2] = np.sim(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        # padding的位置需要单独配置
        pad_row = torch.zeros(1, model_dim)
        position_encoding = torch.cat((pad_row, position_encoding), 0)
        # 构造不可学习的position embedding 参数（这里的posiiton embedding 是绝对位置编码）
        # 总长度是：max_seq_len + 1
        self.position_embedding = nn.Embedding(max_seq_len + 1, model_dim)
        self.position_embedding.weight = nn.Parameter(position_encoding, requires_grad=False)
    
    def forward(self, input_len):
        """
        输入序列，得到对应的位置的 position embedding
        """
        # 找到 input_len 中最大的数（意味着长度）
        max_len = torch.max(input_len)
        # 确定tensor 类型
        tensor = torch.cuda.LongTensor if input_len.is_cuda else  torch.LongTensor
        input_pos = tensor(
           [list(range(1, 1 + seq) + [0] * max_len - seq) for seq in input_len]
        )
        return self.position_embedding(input_pos)

class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalEmbedding, self).__init__()


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