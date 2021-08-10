import torch
import torch.nn as nn
import math

# padding mask
def padding_mask(seq_q, seq_k):
    batch_szie, q_len = seq_q.size()
    mask = seq_k.data.eq(0)
    mask = mask.unsqueeze(1).expand(-1, q_len, -1)
    return mask

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf( x / math.sqrt(2.0)))


class Embedding(nn.Module):

    def __init__(self, vocab_size, max_seq_len, model_dim, segments_num=2):
        super(Embedding, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, model_dim)
        self.seg_embedding = nn.Embedding(segments_num, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)
    
    def forward(self, x, seq):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)
        emebdding = self.word_embedding(x) + self.pos_embedding(pos) + self.seg_embedding(seq)
        return self.layer_norm(emebdding)


class ScaledDotProductAttention(nn.Module):

    def __init__(self, model_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.model_dim = model_dim
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, q, k, v, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1))/ math.sqrt(self.model_dim)
        if attn_mask is not None:
            scores = scores.masked_fill_(attn_mask, -1e-9)
        attention = self.softmax(scores)
        context = torch.matmul(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=768, heads_num=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.per_head_dim = model_dim // heads_num
        self.heads_num = heads_num
        self.linear_q = nn.Linear(model_dim, model_dim)
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)
        self.final_linear = nn.Linear(model_dim, model_dim)
        self.scaled_dot_product_attention = ScaledDotProductAttention(model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)
    
    def forward(self, query, key, value, attn_mask=None):
        residual = query
        batch_size = query.size(0)
        Q = self.linear_q(query).view(batch_size, -1, self.heads_num, self.per_head_dim).transpose(1, 2)
        K = self.linear_k(key).view(batch_size, -1, self.heads_num, self.per_head_dim).transpose(1, 2)
        V = self.linear_v(value).view(batch_size, -1, self.heads_num, self.per_head_dim).transpose(1, 2)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.heads_num, 1, 1)
        context, attention = self.scaled_dot_product_attention(Q, K, V, attn_mask)
        
        # context 需要concate
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.heads_num * self.per_head_dim)
        output = self.final_linear(context)
        output = self.layer_norm(residual + output)
        return output, attention


class PositionWiseFeedForward(nn.Module):

    def __init__(self, model_dim=768, ffn_dim=3072):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)
    
    def forward(self, x):
        output = gelu(self.fc1(x))
        output = self.fc2(output)
        return self.layer_norm(x + output)


class EncoderLayer(nn.Module):

    def __init__(self, model_dim=768, heads_num=8, ffn_dim=3072, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, heads_num, dropout)
        self.feed_forward = PositionWiseFeedForward(model_dim, ffn_dim)
    
    def forward(self, encoder_inputs, attn_mask=None):
        encoder_output, _ = self.attention(encoder_inputs, encoder_inputs, encoder_inputs, attn_mask)
        encoder_output = self.feed_forward(encoder_output)
        return encoder_output


class BertPooler(nn.Module):

    def __init__(self, model_dim=768):
        super(BertPooler, self).__init__()
        self.fc = nn.Linear(model_dim, model_dim)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_state):
        first_token_tensor = hidden_state[:, 0]
        pooled_output = self.fc(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertForMLMPreditionHead(nn.Module):
    
    def __init__(self,model_dim=768):
        super(BertForMLMPreditionHead, self).__init__()
        self.dense = nn.Linear(model_dim, model_dim)
        self.activation = gelu
    
    def forward(self, encoder_output, masked_pos):
        model_dim = encoder_output.size(-1)
        masked_pos = masked_pos.unsqueeze(-1).expand(-1, -1, model_dim)
        h_mask = torch.gather(encoder_output, 1, masked_pos)
        h_mask = self.activation(self.dense(h_mask))
        return h_mask


class BERT(nn.Module):

    def __init__(self, vocab_size, max_seq_len, model_dim=768, heads_num=8, ffn_dim=3072, layers_num=12, dropout=0.2):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size, max_seq_len, model_dim)
        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, heads_num, ffn_dim, dropout) for _ in range(layers_num)])
        self.bert_pooler = BertPooler(model_dim)
        self.bert_masked = BertForMLMPreditionHead(model_dim)
        # NSP 的分类任务
        self.dense_classifier = nn.Linear(model_dim, 2)
        # MLM 的分类任务
        self.dense_lm = nn.Linear(model_dim, vocab_size, bias=False)
    
    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        attn_mask = padding_mask(input_ids, input_ids)
        for encoder_layer in self.encoder_layers:
            output = encoder_layer(output, attn_mask)
        # 分类NSP任务
        h_pooled = self.bert_pooler(output)
        logits_clsf = self.dense_classifier(h_pooled)
        # MLM 任务
        h_masked = self.bert_masked(output, masked_pos)
        logits_lm = self.dense_lm(h_masked)
        return logits_clsf, logits_lm