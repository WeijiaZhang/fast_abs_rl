import os
import sys

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .extract import INI
from .extract import ConvSentEncoder, LSTMEncoder, LSTMPointerNet
from .rnn import MultiLayerLSTMCells
from .util import len_mask
from .attention import prob_normalize


class RerankExtractor(LSTMPointerNet):
    """Revised Pointer network as in Vinyals et al """

    def __init__(self, input_dim, n_classes, emb_dim_cls, n_hidden, n_layer,
                 dropout, n_hop):
        super().__init__(input_dim, n_hidden, n_layer, dropout, n_hop)

        self._n_classes = n_classes
        self._embedding_cls = nn.Embedding(
            n_classes + 1, emb_dim_cls, padding_idx=n_classes)
        self._rerank_v = nn.Parameter(torch.Tensor(n_hidden, n_classes))
        self._stop_linear = nn.Linear(n_hidden, 1)

        # self._cls_wm = nn.Parameter(torch.Tensor(emb_dim_cls, n_hidden))
        self._cls_wq = nn.Parameter(torch.Tensor(input_dim, emb_dim_cls))
        self._cls_v = nn.Parameter(torch.Tensor(emb_dim_cls))

        init.xavier_normal_(self._rerank_v)
        init.xavier_normal_(self._embedding_cls.weight)

        # init.xavier_normal_(self._cls_wm)
        init.xavier_normal_(self._cls_wq)
        init.uniform_(self._cls_v, -INI, INI)
        # convolution for ptr
        # self._conv_ptr = nn.Conv1d(
        #     input_dim, input_dim, kernel_size=3, padding=1)

    def forward(self, attn_mem, tar_in, mem_sizes_src, mem_sizes_tgt):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""
        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem)
        lstm_in = self._get_lstm_in(
            attn_mem, tar_in, mem_sizes_src, mem_sizes_tgt, self._n_classes)

        lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1)
        query, final_states = self._lstm(lstm_in, lstm_states)
        query = query.transpose(0, 1)

        for _ in range(self._n_hop):
            query = LSTMPointerNet.attention(
                hop_feat, query, self._hop_v, self._hop_wq, mem_sizes_src)

        output_stop = self._stop_linear(query).squeeze(-1)
        output = RerankExtractor.rerank_score(
            attn_feat, query, self._rerank_v, self._attn_wq)

        # import pdb
        # pdb.set_trace()

        return output, output_stop  # unormalized extraction logit

    def extract(self, attn_mem, mem_sizes, max_dec_step, thre):
        """extract k sentences, decode only, batch_size==1"""
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        cls_feat = torch.matmul(attn_mem, self._cls_wq.unsqueeze(0))
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        rerank_list = []
        for _ in range(max_dec_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)

            output_stop = torch.sigmoid(self._stop_linear(query)).item()
            if output_stop > thre:
                break
            output = RerankExtractor.rerank_score(
                attn_feat, query, self._rerank_v, self._attn_wq)
            score = F.softmax(output, dim=-1)
            rerank_cls = score.max(dim=-1)[1]
            rerank_cls_emb = self._embedding_cls(rerank_cls)
            output_cls = RerankExtractor.class_score(
                cls_feat, rerank_cls_emb, self._cls_v)
            # for e in extracts:
            #     score[e] = -1e6
            lstm_states = (h, c)
            score_cls = F.softmax(output_cls, dim=-1)
            lstm_in = torch.matmul(score_cls, attn_mem).squeeze(1)
            rerank_list.append(rerank_cls.view(-1).tolist())
        return rerank_list

    @staticmethod
    def rerank_score(attention, query, v, w):
        """ unnormalized rerank score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)
        ).unsqueeze(2)  # [B, Nq, Ns, D]
        score = torch.matmul(
            torch.tanh(sum_), v.unsqueeze(0).unsqueeze(1)
        )  # [B, Nq, Ns, C]
        return score

    @staticmethod
    def class_score(attention, class_emb, v):
        """ unnormalized class score"""
        sum_ = attention.unsqueeze(1) + class_emb   # [B, Nq, Ns, Dc]
        score = torch.matmul(
            torch.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]
        return score

    def _pad_target(self, inputs, max_src_num, max_tgt_num, device, num_pad):
        """pad_batch_multi_step

        :param inputs: List of size B containing list of size T (three-level lists)
        :type inputs: List[List[List]]
        :rtype: TorchTensor of size (B, T, ...)
        """
        batch_size = len(inputs)
        # adding stop control
        tensor_shape = (batch_size, max_tgt_num, max_src_num)
        res_tensor = torch.LongTensor(*tensor_shape).to(device)
        res_tensor.fill_(num_pad)
        for i, ids in enumerate(inputs):
            for j, ele in enumerate(ids):
                ele_tensor = torch.LongTensor(ele).to(device)
                res_tensor[i, j, :len(ele)] = ele_tensor
        return res_tensor

    def _get_lstm_in(self, attn_mem, tar_in, mem_sizes_src, mem_sizes_tgt, num_pad):
        """multi extract k sentences"""
        max_src_num = attn_mem.size(1)
        max_tgt_num = max(mem_sizes_tgt)
        tar_in_pad = self._pad_target(
            tar_in, max_src_num, max_tgt_num, attn_mem.device, num_pad)

        tar_in_emb = self._embedding_cls(tar_in_pad)
        cls_feat = torch.matmul(attn_mem, self._cls_wq.unsqueeze(0))

        score_cls = RerankExtractor.class_score(
            cls_feat, tar_in_emb, self._cls_v)

        mask = len_mask(mem_sizes_src, score_cls.device).unsqueeze(-2)
        norm_score = prob_normalize(score_cls, mask)
        output = torch.matmul(norm_score, attn_mem)
        # import pdb
        # pdb.set_trace()
        return output


class RerankExtractSumm(nn.Module):
    """ multi-ext """

    def __init__(self, emb_dim, vocab_size, n_classes, emb_dim_cls, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 n_hop=1, dropout=0.0):
        super().__init__()
        self._n_classes = n_classes
        self.emb_dim_cls = emb_dim_cls
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout)
        self._art_enc = LSTMEncoder(
            3 * conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self._extractor = RerankExtractor(
            enc_out_dim, n_classes, emb_dim_cls, lstm_hidden, lstm_layer,
            dropout, n_hop
        )

    def forward(self, article_sents, tar_in, sent_nums_src, sent_nums_tgt):
        max_src_num = max(sent_nums_src)
        enc_out = self._encode(
            article_sents, sent_nums_src, max_n=max_src_num)

        output, output_stop = self._extractor(
            enc_out, tar_in, sent_nums_src, sent_nums_tgt)

        out_no_pad, out_stop_no_pad = [], []
        for i, (n_src, n_tgt) in enumerate(zip(sent_nums_src, sent_nums_tgt)):
            out_no_pad.append(
                output[i, :n_tgt, :n_src].contiguous())
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
        # if sent_nums is None:
        #     lstm_out_pad = lstm_out
        # else:
        #     lstm_out_pad = torch.cat([lstm_out, torch.zeros(
        #         lstm_out.size(0), 1, lstm_out.size(2)).to(lstm_out.device)], dim=1)
        return lstm_out

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)
