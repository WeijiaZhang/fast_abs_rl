import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .extract import ConvSentEncoder
from .copy_summ import CopySumm


class RerankCopySumm(nn.Module):
    def __init__(self, vocab_size, emb_dim, conv_hidden,
                 n_hidden, bidirectional, n_layer, dropout=0.0):
        super().__init__()
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout)
        self._copy_sum = CopySumm(vocab_size, emb_dim,
                                  n_hidden, bidirectional, n_layer, dropout)

    def forward(self, article, art_lens, abstract, extend_art, extend_vsize):
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        logit = self._decoder(
            (attention, mask, extend_art, extend_vsize),
            init_dec_states, abstract
        )
        return logit

    def batch_decode(self, article, art_lens, extend_art, extend_vsize,
                     go, eos, unk, max_len):
        """ greedy decode support batching"""
        outputs, attns = self.batch_decode(article, art_lens, extend_art, extend_vsize,
                                           go, eos, unk, max_len)
        return outputs, attns

    def decode(self, article, extend_art, extend_vsize, go, eos, unk, max_len):
        outputs, attns = self._copy_sum.decode(
            article, extend_art, extend_vsize, go, eos, unk, max_len)
        return outputs, attns

    def batched_beamsearch(self, article, art_lens,
                           extend_art, extend_vsize,
                           go, eos, unk, max_len, beam_size, diverse=1.0):

        outputs = self._copy_sum(article, art_lens,
                                 extend_art, extend_vsize,
                                 go, eos, unk, max_len, beam_size, diverse=1.0)
        return outputs
