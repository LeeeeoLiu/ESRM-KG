"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn

import onmt
from onmt.encoders.encoder import EncoderBase
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.rnn_factory import rnn_factory
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F
import torch.nn.init as weigth_init


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = onmt.modules.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
            * all_attn `[batch_size x src_len x src_len]`
        """
        input_norm = self.layer_norm(inputs)
        context, all_attn = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out), all_attn


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, ablation=None):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.emb_size = embeddings.embedding_size
        self.ablation = ablation

        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = dropout

        self.global_gru = nn.GRU(input_size=self.emb_size, hidden_size=d_model, num_layers=num_layers, dropout=dropout, bidirectional=False)
        for weight in self.global_gru.parameters():
            if len(weight.size()) > 1:
                weigth_init.orthogonal(weight.data)
        self._initialize_bridge('GRU', d_model, num_layers)

        if self.ablation is not None:
            _hybrid_dim = d_model
        else:
            _hybrid_dim = d_model*2

        self.linear_trans_hybird = nn.Sequential(
            nn.Linear(_hybrid_dim, d_model),
            nn.LeakyReLU(),
            nn.LayerNorm(d_model, eps=1e-6),
        )

    def forward(self, src, lengths=None, input_weight=0, last_weight=0):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        # src_len X batch_size X emb_dim
        emb = self.embeddings(src)

        # batch_size X src_len X emb_dim
        out = emb.transpose(0, 1).contiguous()

        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)

        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out, _attn = self.transformer[i](out, mask)
        out = self.layer_norm(out)

        # memory_bank src_len x batch_size x d_model
        memory_bank = out.transpose(0, 1).contiguous()

        packed_emb = emb
        if lengths is not None:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)

        # Global Encoder
        _, hidden_g = self.global_gru(packed_emb)
        hidden_g = self._bridge(hidden_g)

        # local_encoded : batch_size x 1 x d_model
        _len = memory_bank.size(0)
        tf_local_encoded = torch.bmm(_attn, memory_bank.permute(1, 0, 2)).sum(1).unsqueeze(1)/_len

        # global_encoded : batch_size x 1 x d_model
        global_encoded = hidden_g[-1].unsqueeze(0).permute(1, 0, 2)

        if self.ablation is None:
            hybrid_encoded_list = [global_encoded, tf_local_encoded]
        elif self.ablation == "Global":
            hybrid_encoded_list = [tf_local_encoded]
        elif self.ablation == "Transformer":
            hybrid_encoded_list = [global_encoded]
        else:
            raise NotImplementedError

        hybrid_encoded = torch.cat(hybrid_encoded_list, 2)

        # mul the attention : batch_szie x 1 x d_model
        _encoded = self.linear_trans_hybird(hybrid_encoded)

        # batch_size x 1 x emb_size
        emb_vocab_size = self.embeddings.make_embedding.emb_luts[0].num_embeddings

        encoded = torch.bmm(_encoded, self.embeddings(torch.LongTensor([_idx for _idx in range(
            emb_vocab_size)]).repeat(src.size(1), 1).permute(1, 0).unsqueeze(2).to(src.device)).permute(1, 2, 0))

        if input_weight > 0:
            _log_softmax = torch.nn.LogSoftmax(dim=-1)
            encoded = _log_softmax(encoded.squeeze())
            _src = src.squeeze().permute(1, 0)
            _input_attn = torch.zeros(encoded.size()).to(encoded.device)
            for _b in range(encoded.size(0)):
                _idx_list = _src[_b].tolist()
                # remove padding index
                _idx_list = [_idx for _idx in _idx_list if _idx != 1]
                _input_attn[_b, [_idx_list]] += input_weight
                if last_weight > 0:
                    _input_attn[_b, _idx_list[-1]] += last_weight
            encoded = (encoded + _input_attn).unsqueeze(1)

        return hidden_g, memory_bank, lengths, encoded


    def _initialize_bridge(self, rnn_type, hidden_size, num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.leaky_relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs
