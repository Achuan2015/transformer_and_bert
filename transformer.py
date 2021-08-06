import torch.nn as nn
from util import *


class Transformer(nn.Module):

    def __init__(self, vocab_size, max_seq_len, model_dim=512, num_heads=8, num_layers=6, ffn_dim=2048, dropout=0.0):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size, max_seq_len, model_dim, num_layers, num_heads, ffn_dim, dropout)
        self.decoder = Decoder(vocab_size, max_seq_len, model_dim, num_heads, num_layers, ffn_dim, dropout)
        # decoder module之后还有一个 linaer layer 和 softmax layer
        self.linear = nn.Linear(model_dim, model_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        output, enc_self_attn = self.encoder(src_seq, src_len)
        output, self_attn_attentions, context_attn_attentions = self.decoder(tgt_seq, tgt_len, output)
        output = self.linear(output)
        output = self.softmax(output)
        return output, enc_self_attn, self_attn_attentions, context_attn_attentions