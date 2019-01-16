""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import argparse
import json
import os
from os.path import join
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op

from cytoolz import identity, concat, curry

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp

from data.batcher import tokenize

from decoding import Abstractor, RLExtractor, DecodeDataset, AnalysisDataset, BeamAbstractor
from decoding import make_html_safe
from evaluate import eval_rouge_by_cmd


def format_rouge(summaries, references, split='test'):
    results_dict = eval_rouge_by_cmd(summaries, references, split=split)
    res = []
    for idx in ['1', '2', 'l']:
        head_prefix = "ROUGE-%s" % idx.upper()
        val_prefix = "rouge_%s" % idx
        temp = "{}-R: {:.2f}, {}-P: {:.2f}, {}-F: {:.2f}".format(head_prefix, results_dict["%s_recall" % val_prefix] * 100,
                                                                 head_prefix, results_dict["%s_precision" %
                                                                                           val_prefix] * 100,
                                                                 head_prefix, results_dict["%s_f_score" % val_prefix] * 100)
        res.append(temp)
    return res


def decode(save_path, model_dir, split, batch_size,
           beam_size, diverse, max_len, cuda):
    start = time()
    # setup model
    with open(join(model_dir, 'meta.json')) as f:
        meta = json.loads(f.read())
    if meta['net_args']['abstractor'] is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        assert beam_size == 1
        abstractor = identity
    else:
        if beam_size == 1:
            abstractor = Abstractor(join(model_dir, 'abstractor'),
                                    max_len, cuda)
        else:
            abstractor = BeamAbstractor(join(model_dir, 'abstractor'),
                                        max_len, cuda)
    extractor = RLExtractor(model_dir, cuda=cuda)

    # setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles
    dataset = DecodeDataset(split)
    analysis_dataset = AnalysisDataset(split)

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )

    # prepare save paths and logs
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    dec_path = join(save_path, split)
    if not os.path.isdir(dec_path):
        os.makedirs(dec_path)
    dec_log = {}
    dec_log['abstractor'] = meta['net_args']['abstractor']
    dec_log['extractor'] = meta['net_args']['extractor']
    dec_log['rl'] = True
    dec_log['split'] = split
    dec_log['beam'] = beam_size
    dec_log['diverse'] = diverse
    with open(join(save_path, 'log_%s.json' % split), 'w') as f:
        json.dump(dec_log, f, indent=4)

    # Decoding
    i = 0
    is_top5_all = []
    with torch.no_grad():
        for i_debug, raw_article_batch in enumerate(loader):
            tokenized_article_batch = map(tokenize(None), raw_article_batch)
            ext_arts = []
            ext_inds = []
            ext_preds = []
            for raw_art_sents in tokenized_article_batch:
                ext = extractor(raw_art_sents)[:-1]  # exclude EOE
                if not ext:
                    # use top-5 if nothing is extracted
                    # in some rare cases rnn-ext does not extract at all
                    ext = list(range(5))[:len(raw_art_sents)]
                    is_top5_all.append(True)
                else:
                    ext = [i.item() for i in ext]
                    is_top5_all.append(False)
                ext_inds += [(len(ext_arts), len(ext))]
                ext_preds += ext
                ext_arts += [raw_art_sents[i] for i in ext]
            if beam_size > 1:
                all_beams = abstractor(ext_arts, beam_size, diverse)
                dec_outs = rerank_mp(all_beams, ext_inds)
            else:
                dec_outs = abstractor(ext_arts)
            assert i == batch_size * i_debug
            for j, n in ext_inds:
                out_data = {}
                js_data = analysis_dataset[i]
                decoded_sents = [make_html_safe(
                    ' '.join(dec)) for dec in dec_outs[j:j + n]]
                out_data['id'] = js_data['id']
                out_data['article'] = js_data['article']
                out_data['abstract'] = js_data['abstract']
                out_data['decode'] = decoded_sents
                if split == 'val':
                    out_data['extracted'] = js_data['extracted']
                out_data['extract_preds'] = ext_preds[j:j + n]
                # calculating rouge score
                rouge_str = format_rouge(
                    [decoded_sents], [out_data['abstract']], split=split)
                out_data['rouge'] = rouge_str
                out_data['is_top'] = is_top5_all[i]
                if split == 'val':
                    out_data['score'] = js_data['score']
                with open(join(save_path, split, '{}.json'.format(i)),
                          'w') as f:
                        # f.write(('\n'.join(decoded_sents)))
                    json.dump(out_data, f, indent=4)
                i += 1
                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                    i, n_data, i / n_data * 100,
                    timedelta(seconds=int(time() - start))
                ), end='')
                # import pdb
                # pdb.set_trace()

    print("\nThe Number of Top5: %d" % sum(is_top5_all))


_PRUNE = defaultdict(
    lambda: 2,
    {1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 4, 7: 3, 8: 3}
)


def rerank(all_beams, ext_inds):
    beam_lists = (all_beams[i: i + n] for i, n in ext_inds if n > 0)
    return list(concat(map(rerank_one, beam_lists)))


def rerank_mp(all_beams, ext_inds):
    beam_lists = [all_beams[i: i + n] for i, n in ext_inds if n > 0]
    with mp.Pool(8) as pool:
        reranked = pool.map(rerank_one, beam_lists)
    return list(concat(reranked))


def rerank_one(beams):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)
    dec_outs = [h.sequence for h in best_hyps]
    return dec_outs


def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i + n]) for i in range(len(sequence) - (n - 1)))


def _compute_score(hyps):
    all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    repeat = sum(c - 1 for g, c in all_cnt.items() if c > 1)
    lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    return (-repeat, lp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run decoding of the full model (RL)')
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--model_dir', help='root of the full model')

    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--val', action='store_true', help='use validation set')
    data.add_argument('--test', action='store_true', help='use test set')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--beam', type=int, action='store', default=1,
                        help='beam size for beam-search (reranking included)')
    parser.add_argument('--div', type=float, action='store', default=1.0,
                        help='diverse ratio for the diverse beam-search')
    parser.add_argument('--max_dec_word', type=int, action='store', default=30,
                        help='maximun words to be decoded for the abstractor')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    data_split = 'test' if args.test else 'val'
    decode(args.path, args.model_dir,
           data_split, args.batch, args.beam, args.div,
           args.max_dec_word, args.cuda)
