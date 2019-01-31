import os
import sys

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .extract import ConvSentEncoder, LSTMEncoder, LSTMPointerNet


class MultiStepDecoder(LSTMPointerNet):
    """Revised Pointer network as in Vinyals et al """

    def __init__(self, input_dim, n_hidden, n_layer,
                 dropout, n_hop):
        super().__init__(input_dim, n_hidden, n_layer, dropout, n_hop)
        self._stop_linear = nn.Linear(n_hidden, 1)
        # convolution for ptr
        # self._conv_ptr = nn.Conv1d(
        #     input_dim, input_dim, kernel_size=3, padding=1)

    def forward(self, attn_mem_pad, tar_in, mem_sizes):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""
        attn_mem, lstm_in = self._get_lstm_in(attn_mem_pad, tar_in)
        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem)
        lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1)
        query, final_states = self._lstm(lstm_in, lstm_states)
        query = query.transpose(0, 1)

        for _ in range(self._n_hop):
            query = LSTMPointerNet.attention(
                hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)

        output_stop = self._stop_linear(query).squeeze(-1)
        output = LSTMPointerNet.attention_score(
            attn_feat, query, self._attn_v, self._attn_wq)

        return output, output_stop  # unormalized extraction logit

    def extract(self, attn_mem, mem_sizes, max_dec_step, thre):
        """extract k sentences, decode only, batch_size==1"""
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []
        for _ in range(max_dec_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)

            output_stop = torch.sigmoid(self._stop_linear(query)).item()
            if len(extracts) > 0 and output_stop > thre:
                break
            output = LSTMPointerNet.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            score = torch.sigmoid(output.view(-1)).tolist()
            # for e in extracts:
            #     score[e] = -1e6
            ext = []
            for i, s in enumerate(score):
                if s > thre:
                    ext.append(i)
            extracts.append(ext)
            lstm_states = (h, c)
            ext_idx = torch.LongTensor(ext)
            lstm_in = attn_mem[:, ext_idx, :].mean(dim=1)
        return extracts

    def _pad_target(self, inputs, pad, device):
        """pad_batch_multi_step

        :param inputs: List of size B containing list of size T (three-level lists)
        :type inputs: List[List[List]]
        :rtype: TorchTensor of size (B, T, ...)
        """
        batch_size = len(inputs)
        max_len = 0
        max_len_one_step = 0
        for ids in inputs:
            max_len = max(len(ids), max_len)
            max_len_one_step = max(max(map(len, ids)), max_len_one_step)

        # adding stop control
        tensor_shape = (batch_size, max_len, max_len_one_step)
        res_tensor = torch.LongTensor(*tensor_shape).to(device)
        res_tensor.fill_(pad)
        for i, ids in enumerate(inputs):
            for j, ele in enumerate(ids):
                ele_tensor = torch.LongTensor(ele).to(device)
                res_tensor[i, j, :len(ele)] = ele_tensor
        return res_tensor

    def _get_lstm_in(self, enc_out_pad, tar_in):
        """multi extract k sentences"""
        num_pad = enc_out_pad.size(1) - 1
        tar_in_pad = self._pad_target(tar_in, num_pad, enc_out_pad.device)
        bs, nt, nt_e = tar_in_pad.size()
        d = enc_out_pad.size(2)
        ptr_in = []
        for i in range(nt):
            target = tar_in_pad[:, i, :]
            seq_lens = list(
                map(lambda x: len(x[i]) if i < len(x) else 0, tar_in))
            ptr_one = torch.gather(
                enc_out_pad, dim=1, index=target.unsqueeze(2).expand(bs, nt_e, d)
            )
            ptr_one_sum = ptr_one.sum(dim=1)
            ptr_one_mean = torch.stack(
                [s / l if l > 0 else s for s, l in zip(ptr_one_sum, seq_lens)], dim=0)
            ptr_in.append(ptr_one_mean)

        enc_out = enc_out_pad[:, :-1, :]
        ptr_in = torch.stack(ptr_in, dim=1)
        return enc_out, ptr_in


class MultiExtractSumm(nn.Module):
    """ multi-ext """

    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 n_hop=1, dropout=0.0):
        super().__init__()
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout)
        self._art_enc = LSTMEncoder(
            3 * conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self._extractor = MultiStepDecoder(
            enc_out_dim, lstm_hidden, lstm_layer,
            dropout, n_hop
        )

    def forward(self, article_sents, tar_in, sent_nums_src, sent_nums_tgt):
        num_pad = max(sent_nums_src)
        enc_out_pad = self._encode(
            article_sents, sent_nums_src, max_n=num_pad)

        output, output_stop = self._extractor(
            enc_out_pad, tar_in, sent_nums_src)

        out_no_pad, out_stop_no_pad = [], []
        for i, (n_src, n_tgt) in enumerate(zip(sent_nums_src, sent_nums_tgt)):
            out_no_pad.append(output[i, :n_tgt, :n_src].contiguous().view(-1))
            out_stop_no_pad.append(
                output_stop[i, :(n_tgt + 1)].contiguous().view(-1))

        return out_no_pad, out_stop_no_pad

    def extract(self, article_sents, sent_nums=None, max_dec_step=10, thre=0.5):
        enc_out = self._encode(article_sents, sent_nums)
        output = self._extractor.extract(
            enc_out, sent_nums, max_dec_step, thre=thre)
        return output

    def _encode(self, article_sents, sent_nums, max_n=1):
        if sent_nums is None:  # test-time excode only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            # +1 means to add padding idx
            enc_sents = [self._sent_enc(art_sent)
                         for art_sent in article_sents]

            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z

            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n - n, s.device)], dim=0)
                 if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            )

        lstm_out = self._art_enc(enc_sent, sent_nums)
        lstm_out_pad = torch.cat([lstm_out, torch.zeros(
            lstm_out.size(0), 1, lstm_out.size(2)).to(lstm_out.device)], dim=1)
        return lstm_out_pad

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)
