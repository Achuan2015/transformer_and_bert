from transformer import PositionalEmbedding
import torch
from transformer_recurrent import *

batch_size = 10
max_seq_len = 512
model_dim = 512
num_heads = 8
dropout = 0.1

def test_ScaledDotProductAttention():
    q = torch.randn(batch_size, max_seq_len, model_dim)
    k = torch.randn(batch_size, max_seq_len, model_dim)
    v = torch.randn(batch_size, max_seq_len, model_dim)
    func = ScaledDotProductAttention(0.1)
    context, attn = func(q, k, v)
    assert context.size() == (batch_size, max_seq_len, model_dim)
    assert attn.size() == (batch_size, max_seq_len, max_seq_len)
    print('test_ScaledDotProductAttention pass')

def test_MultiHeadAttention():
    q = torch.randn(batch_size, max_seq_len, model_dim)
    k = torch.randn(batch_size, max_seq_len, model_dim)
    v = torch.randn(batch_size, max_seq_len, model_dim)
    func = MultiHeadAttention(model_dim, num_heads, dropout)
    output, attention = func(k, v, q)
    assert output.size() == (batch_size, max_seq_len, model_dim)
    assert attention.size() == (batch_size * num_heads, max_seq_len, max_seq_len)
    print('test_MultiHeadAttention pass')

def test_padding_mask():
    input_q = torch.Tensor([[1,2,3,23]])
    input_k = torch.Tensor([[2,32,1,1]])
    mask = padding_mask(input_q, input_k)
    assert mask.shape == (1, 4, 4)
    print('test_padding_mask pass')

def test_sequence_mask():
    input_q = torch.Tensor([[1,2,3,23], [2,32,1,1], [2,32,1,1]])
    mask = sequence_mask(input_q)
    assert mask.shape == (3, 4, 4)
    print('test_sequence_mask pass')

def test_PositionalEncoding():
    input_len = [10, 29, 23, 12]
    func = PositionalEncoding(model_dim, max_seq_len)
    emb = func(input_len)
    assert emb.shape == (4, max_seq_len, model_dim)
    print('test_PositionalEncoding pass')

if __name__ == '__main__':
    # test_ScaledDotProductAttention()
    # test_MultiHeadAttention()
    # test_padding_mask()
    # test_sequence_mask()
    test_PositionalEncoding()