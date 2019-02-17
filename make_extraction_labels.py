"""produce the dataset with (psudo) extraction label"""
import os
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import multiprocessing as mp

from cytoolz import curry, compose
from nltk.stem import SnowballStemmer
from heapq import nlargest

from utils import MAX_SENT_ART
from utils import count_data
from metric import compute_rouge_n, compute_rouge_l, compute_rouge_l_summ


# try:
#     DATA_DIR = os.environ['DATA']
# except KeyError:
#     print('please use environment variable to specify data directories')
# DATA_DIR = '/home/zhangwj/code/nlp/summarization/dataset/raw/CNN_Daily/fast_abs_rl/finished_files'
# DATA_DIR = '/home/zhangwj/code/nlp/summarization/fast_abs_rl/output/data_analysis/multi_extract'
# DATA_DIR = '/home/zhangwj/code/nlp/summarization/fast_abs_rl/output/data_analysis/max_rouge_l'
DATA_DIR = '/home/zhangwj/code/nlp/summarization/dataset/raw/CNN_Daily/rerank'


def _split_words(texts):
    return map(lambda t: t.split(), texts)


def get_extract_label(art_sents, abs_sents):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    scores = []
    indices = list(range(len(art_sents)))
    for abst in abs_sents:
        rouges = list(map(compute_rouge_l(reference=abst, mode='r'),
                          art_sents))
        ext = max(indices, key=lambda i: rouges[i])
        indices.remove(ext)
        extracted.append(ext)
        scores.append(rouges[ext])
        if not indices:
            break
    return extracted, scores


def get_stemming(sentence):
    res = list(map(lambda x: stemmer.stem(x), sentence))
    return res


def get_multi_extract(art_sents, abs_sents, top_n=3, mode='r', weights=[0.3, 0.2, 0.5]):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    scores_n1, scores_n2 = [], []
    scores_l, scores_w = [], []
    art_sents_stem = list(map(get_stemming, art_sents))
    for abst in abs_sents:
        indices = list(range(len(art_sents)))
        abst_stem = get_stemming(abst)
        rouges_n1 = list(map(compute_rouge_n(reference=abst_stem, n=1, mode=mode),
                             art_sents_stem))
        rouges_n2 = list(map(compute_rouge_n(reference=abst_stem, n=2, mode=mode),
                             art_sents_stem))
        rouges_l = list(map(compute_rouge_l(reference=abst_stem, mode=mode),
                            art_sents_stem))
        rouges_w = []
        for r_n1, r_n2, r_l in zip(rouges_n1, rouges_n2, rouges_l):
            r_w = weights[0] * r_n1 + weights[1] * r_n2 + weights[2] * r_l
            rouges_w.append(r_w)
        # ext = max(indices, key=lambda i: rouges[i])
        ext_idx = nlargest(top_n, indices, key=lambda i: rouges_w[i])
        extracted.append(", ".join(map(str, ext_idx)))
        scores_n1.append(", ".join([str(rouges_n1[idx]) for idx in ext_idx]))
        scores_n2.append(", ".join([str(rouges_n2[idx]) for idx in ext_idx]))
        scores_l.append(", ".join([str(rouges_l[idx]) for idx in ext_idx]))
        scores_w.append(", ".join([str(rouges_w[idx]) for idx in ext_idx]))

    scores = (scores_n1, scores_n2, scores_l, scores_w)
    return extracted, scores


def get_max_rouge_l(art_sents, abs_sents, mode='r', thre=0.69):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    scores_l = []
    score_l_max = []
    art_sents_stem = list(map(get_stemming, art_sents))
    for abst in abs_sents:
        indices = list(range(len(art_sents)))
        abst_stem = get_stemming(abst)
        rouges_l = list(map(compute_rouge_l(reference=abst_stem, mode=mode),
                            art_sents_stem))

        # ext = max(indices, key=lambda i: rouges[i])
        indices_sorted = sorted(
            indices, key=lambda i: rouges_l[i], reverse=True)
        init_idx = 0
        max_sent_nums = MAX_SENT_ART - 1
        while init_idx < len(indices_sorted) and indices_sorted[init_idx] > max_sent_nums:
            init_idx += 1

        init_ext_idx = indices_sorted[init_idx]
        ext_idx = [init_ext_idx]
        init_summs = [art_sents_stem[init_ext_idx]]
        refs = [abst_stem]
        init_rouge_l = compute_rouge_l_summ(init_summs, refs, mode=mode)
        for idx in indices_sorted[init_idx + 1:]:
            if idx > max_sent_nums:
                continue
            if (init_rouge_l - thre) > 0.01 and (rouges_l[idx] - 0.49) < 0.01:
                break
            after_summs = init_summs + [art_sents_stem[idx]]
            after_rouge_l = compute_rouge_l_summ(after_summs, refs, mode=mode)
            diff = after_rouge_l - init_rouge_l
            # import pdb
            # pdb.set_trace()
            # introducing label redundancy
            if rouges_l[idx] > 0.59:
                ext_idx.append(idx)
            elif (init_rouge_l > 0.6 and diff > 0.12) or (0.5 < init_rouge_l <= 0.6 and diff > (0.14 - 0.2 * (init_rouge_l - 0.5))) or \
                    (0.4 < init_rouge_l <= 0.5 and diff > (0.18 - 0.4 * (init_rouge_l - 0.4))) or \
                    (0.3 < init_rouge_l <= 0.4 and diff > (0.24 - 0.6 * (init_rouge_l - 0.3))):
                init_rouge_l = after_rouge_l
                init_summs = after_summs
                ext_idx.append(idx)
            else:
                break

        extracted.append(", ".join(map(str, ext_idx)))
        scores_l.append(", ".join([str(rouges_l[idx]) for idx in ext_idx]))
        score_l_max.append(init_rouge_l)

    scores = (scores_l, score_l_max)
    return extracted, scores


def get_rerank(art_sents, abs_sents, mode='r'):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    scores_l = []
    labels_l = []
    # thresholds = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
    #               (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.1)]
    # thresholds = [(0, 0.2), (0.2, 0.3), (0.3, 0.4),
    #               (0.4, 0.5), (0.5, 0.7), (0.7, 1.1)]
    # thresholds = [(0, 0.2), (0.2, 0.25), (0.25, 0.3),
    #               (0.3, 0.35), (0.35, 0.4), (0.4, 0.6), (0.6, 1.1)]
    # thresholds = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.1)]
    thresholds = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.1)]
    art_sents_stem = list(map(get_stemming, art_sents))[:MAX_SENT_ART]
    for abst in abs_sents:
        indices = list(range(len(art_sents_stem)))
        abst_stem = get_stemming(abst)
        rouges_l = list(map(compute_rouge_l(reference=abst_stem, mode=mode),
                            art_sents_stem))

        # ext = max(indices, key=lambda i: rouges[i])
        indices_sorted = sorted(
            indices, key=lambda i: rouges_l[i], reverse=True)
        label = []
        for idx in indices_sorted:
            s = rouges_l[idx] + 0.025
            for i, (low, high) in enumerate(thresholds):
                if (low < s < high) or (s - low < 1e-3) or (s - high < 1e-3):
                    label.append(i)
                    break

        assert len(label) == len(indices_sorted)
        extracted.append(", ".join(map(str, indices_sorted)))
        scores_l.append(", ".join(["%.4f" % rouges_l[idx]
                                   for idx in indices_sorted]))
        labels_l.append(", ".join(map(str, label)))

    scores = (scores_l, labels_l)
    return extracted, scores


@curry
def process(split, i):
    data_dir = join(DATA_DIR, split)
    with open(join(data_dir, '{}.json'.format(i))) as f:
        js_data = json.loads(f.read())

    data = dict()
    data['id'] = js_data['id']
    data['article'] = js_data['article']
    data['abstract'] = js_data['abstract']
    tokenize = compose(list, _split_words)
    art_sents = tokenize(data['article'])
    abs_sents = tokenize(data['abstract'])
    if art_sents and abs_sents:  # some data contains empty article/abstract
        # extracted, scores = get_extract_label(art_sents, abs_sents)
        # extracted, scores = get_multi_extract(art_sents, abs_sents)
        # scores_n1, scores_n2, scores_l, scores_w = scores
        extracted, scores = get_rerank(art_sents, abs_sents)
        scores_l, labels_l = scores
    else:
        extracted, scores = [], []
        # scores_n1, scores_n2 = [], []
        # scores_l, scores_w = [], []
        scores_l, labels_l = [], []
    data['extracted'] = extracted
    # data['rouge_w_r'] = scores_w
    data['rouge_l_r'] = scores_l
    data['label_l_r'] = labels_l
    # data['rouge_l_r_max'] = score_l_max
    # data['rouge_n1_r'] = scores_n1
    # data['rouge_n2_r'] = scores_n2
    with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
        json.dump(data, f, indent=4)


def label_mp(split):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    n_data = count_data(data_dir)
    with mp.Pool() as pool:
        list(pool.imap_unordered(process(split),
                                 list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time() - start)))


def label(split):
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    n_data = count_data(data_dir)
    for i in range(n_data):
        print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100 * i / n_data),
              end='')
        with open(join(data_dir, '{}.json'.format(i))) as f:
            js_data = json.loads(f.read())

        data = dict()
        data['id'] = js_data['id']
        data['article'] = js_data['article']
        data['abstract'] = js_data['abstract']
        tokenize = compose(list, _split_words)
        art_sents = tokenize(data['article'])
        abs_sents = tokenize(data['abstract'])

        # extracted, scores = get_extract_label(art_sents, abs_sents)
        # data['extracted'] = extracted
        # data['score'] = scores

        if art_sents and abs_sents:  # some data contains empty article/abstract
            # extracted, scores = get_extract_label(art_sents, abs_sents)
            # extracted, scores = get_multi_extract(art_sents, abs_sents)
            # scores_n1, scores_n2, scores_l, scores_w = scores
            extracted, scores = get_rerank(art_sents, abs_sents)
            scores_l, labels_l = scores
        else:
            extracted, scores = [], []
            # scores_n1, scores_n2 = [], []
            # scores_l, scores_w = [], []
            scores_l, labels_l = [], []

        data['extracted'] = extracted
        # data['rouge_w_r'] = scores_w
        data['rouge_l_r'] = scores_l
        data['label_l_r'] = labels_l
        # data['rouge_l_r_max'] = score_l_max
        # data['rouge_n1_r'] = scores_n1
        # data['rouge_n2_r'] = scores_n2

        with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
            json.dump(data, f, indent=4)
        # import pdb
        # pdb.set_trace()
    print('finished in {}'.format(timedelta(seconds=time() - start)))


def main():
    # split_all = ['val', 'train']
    split_all = ['val']
    for split in split_all:  # no need of extraction label when testing
        # label(split)
        label_mp(split)


if __name__ == '__main__':
    stemmer = SnowballStemmer('english')
    main()
