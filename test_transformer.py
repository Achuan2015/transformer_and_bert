import torch
from model import Transformer
from util import *


batch_size = 10
max_seq_len = 512
model_dim = 512
ffn_dim = 2048
num_heads = 8
num_layers = 6
dropout = 0.1
vocab_size = 22500


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
    input_len = torch.Tensor([10, 29, 23, 12])
    max_len = torch.max(input_len)
    func = PositionalEncoding(model_dim, max_seq_len)
    emb = func(input_len)
    assert emb.shape == (4, max_len, model_dim)
    print('test_PositionalEncoding pass')

def test_PositionWiseFeedForward():
    x = torch.randn(batch_size, max_seq_len, model_dim)
    func = PositionWiseFeedForward(model_dim, ffn_dim, dropout)
    output = func(x)
    assert output.shape == (batch_size, max_seq_len, model_dim)
    print('test_PositionalWiseFeedForward pass')

def test_EncoderLayer():
    x = torch.randn(batch_size, max_seq_len, model_dim)
    func = EncoderLayer(model_dim, num_heads, ffn_dim, dropout)
    output, attention = func(x)
    assert output.shape == (batch_size, max_seq_len, model_dim)
    assert attention.shape == (batch_size * num_heads, max_seq_len, model_dim)
    print('test_EncoderLayer pass')

def test_EmbeddingLayer():
    tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    inputs = tensor([[1,2,3,3,4,0], [12,3,3,4,5,7], [10, 11, 21, 21,11, 0]])
    inputs_len = torch.Tensor([5, 6, 5])
    max_len = torch.max(inputs_len)
    func = EmbeddingLayer(vocab_size, max_seq_len, model_dim)
    output = func(inputs, inputs_len)
    assert output.shape == (3, max_len, model_dim)
    print('test_EmbeddingLayer pass')

def test_Encoder():
    tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    inputs = tensor([[1,2,3,3,4,0], [12,3,3,4,5,7], [10, 11, 21, 21,11, 0]])
    inputs_len = torch.Tensor([5, 6, 5])
    max_len = torch.max(inputs_len)
    func = Encoder(vocab_size, max_seq_len, model_dim)
    output, attention = func(inputs, inputs_len)
    assert output.shape == (3, max_len, model_dim)
    assert len(attention) == num_layers
    print('test_Encoder pass')

def test_DecoderLayer():
    dec_input = torch.randn(batch_size, max_seq_len, model_dim)
    enc_output = torch.randn(batch_size, max_seq_len, model_dim)
    func = DecoderLayer(model_dim, num_heads, ffn_dim, dropout)
    output, _, _ = func(dec_input, enc_output)
    assert output.shape == (batch_size, max_seq_len, model_dim)
    print('test_DecoderLayer pass')

def test_Decoder():
    tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    inputs = tensor([[1,2,3,3,4,0], [12,3,3,4,5,7], [10, 11, 21, 21,11, 0]])
    inputs_len = torch.Tensor([5, 6, 5])
    enc_output = torch.randn(3, 8, model_dim)
    func = Decoder(vocab_size, max_seq_len, model_dim)
    output, _, _ = func(inputs, inputs_len, enc_output)
    assert output.shape == (3, 6, model_dim)
    print('test_Decoder pass')

def test_Transformer():
    tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    src_inputs = tensor([[1,2,3,3,4,0,0,0], [12,3,3,4,5,7,1,3], [10, 11, 21, 21,11, 0,0,0]])
    src_inputs_len = torch.Tensor([5, 8, 5])
    target_inputs = tensor([[1,2,3,3,4,0], [12,3,3,4,5,7], [10, 11, 21, 21,11, 0]])
    target_inputs_len = torch.Tensor([5, 6, 5])
    model = Transformer(vocab_size, max_seq_len, model_dim, num_heads, num_layers, ffn_dim, dropout)
    output, _, _, _ = model(src_inputs, src_inputs_len, target_inputs, target_inputs_len)
    assert output.shape == (3, 6, model_dim)
    print('test_Transformers pass')

if __name__ == '__main__':
    # test_ScaledDotProductAttention()
    # test_MultiHeadAttention()
    # test_padding_mask()
    # test_sequence_mask()
    # test_PositionalEncoding()
    # test_PositionWiseFeedForward()
    # test_EncoderLayer()
    # test_EmbeddingLayer()
    # test_Encoder()
    # test_DecoderLayer()
    # test_Decoder()
    test_Transformer()