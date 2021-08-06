import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query, key, value, scale=None, attn_mask=None):
        attention = torch.bmm(query, key.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            # print('attn_mask: ', attn_mask.shape)
            # print('attention: ', attention.shape)
            attention = attention.masked_fill_(attn_mask, -np.inf)
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
        if attn_mask is not None:
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
    
    def forward(self, inputs_len):
        # 确定pos embedding 的类型
        inputs_len = inputs_len.to(torch.uint8)
        max_len = torch.max(inputs_len)
        # inputs = [list(range(1, seq_len + 1)) + [0] * (self.max_seq_len - seq_len) for seq_len in input_len]
        inputs = [list(range(1, seq_len + 1)) + [0] * (max_len - seq_len) for seq_len in inputs_len]
        tensor = torch.cuda.LongTensor if inputs_len.is_cuda else torch.LongTensor
        inputs = tensor(inputs)
        return self.position_embedding(inputs)


class PositionWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ffn_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=ffn_dim, out_channels=model_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(model_dim)
    
    def forward(self, x):
        # 因为卷积层是seq_len 这个维度滑动，因此x的维度需要转置；具体进行可以细看 nn.Conv1d 的input说明
        out = x.transpose(1, 2)
        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        output = out.transpose(1, 2)
        # add residual and layernrom
        output = self.layer_norm(x + output)
        return output


class EncoderLayer(nn.Module):
    """
    组合 multi-head attention 模块和 FFN 模块
    """
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(model_dim=model_dim, ffn_dim=ffn_dim, dropout=dropout)
    
    def forward(self, x, attn_mask=None):
        # multi-head attention module
        context, attention = self.attention(x, x, x, attn_mask)
        # feed forward module
        output = self.feed_forward(context)
        return output, attention


class EmbeddingLayer(nn.Module):
    """
    组合word_embedding 和 position_embedding
    """

    def __init__(self, vocab_size, model_dim=512, max_seq_len=512):
        super(EmbeddingLayer, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size+1, model_dim)
        self.positon_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len):
        output = self.word_embedding(inputs)
        output += self.positon_embedding(inputs_len)
        return output


class Encoder(nn.Module):
    
    def __init__(self, vocab_size, max_seq_len, model_dim=512, num_layers=6, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(Encoder, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, max_seq_len, model_dim)
        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
    
    def forward(self, inputs, inputs_len):
        output = self.embedding(inputs, inputs_len)
        attn_mask = padding_mask(inputs, inputs)
        attentions = []
        for encoder_layer in self.encoder_layers:
            output, attention = encoder_layer(output, attn_mask)
            attentions.append(attention)
        return output, attentions

class DecoderLayer(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(model_dim, ffn_dim, dropout)
    
    def forward(self, decoder_input, encoder_output, self_attn_mask=None, context_attn_mask=None):
        # mask multi-head attention module
        dec_output, self_attention = self.attention(decoder_input, decoder_input, decoder_input, self_attn_mask)
        # context multi-head attention module
        dec_output, context_attention = self.attention(encoder_output, encoder_output, dec_output, context_attn_mask)
        # feed forward module
        output = self.feed_forward(dec_output)
        return output, self_attention, context_attention


class Decoder(nn.Module):
    """
    Decoder 组合decoder Layer 与 embedding layer
    这里需要注意的是, context_attn_mask 每次都需要根据decoder的input与inputs 时时计算，因此，在这里 context_attn_mask 仍然以参数的形式传入
    """

    def __init__(self, vocab_size, max_seq_len, model_dim=512, num_heads=8, num_layers=6, ffn_dim=2048, dropout=0.0):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList([DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.embedding_layer = EmbeddingLayer(vocab_size, model_dim, max_seq_len)
    
    def forward(self, inputs, inputs_len, encoder_output, context_attn_mask=None):
        output = self.embedding_layer(inputs, inputs_len)
        
        self_attn_padding_mask = padding_mask(inputs, inputs)
        self_attn_sequence_mask = sequence_mask(inputs)
        self_attn_mask = torch.gt(self_attn_padding_mask + self_attn_sequence_mask, 0)
        
        self_attentions = []
        context_attentions = []
        for decoder_layer in self.decoder_layers:
            output, self_attention, context_attention = decoder_layer(output, encoder_output, self_attn_mask, context_attn_mask)
            self_attentions.append(self_attention)
            context_attentions.append(context_attention)
        return output, self_attentions, context_attentions