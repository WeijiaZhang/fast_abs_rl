""" data analysis """
import argparse
import codecs
import json
import os
import re
from os.path import join, exists
from time import time
from datetime import timedelta
from tqdm import tqdm
from decoding import make_html_safe
import fire
from collections import defaultdict
import numpy as np

from cytoolz import curry, compose, concat
from make_extraction_labels import _split_words
from metric import compute_rouge_l
from evaluate import eval_rouge_by_cmd
from decode_full_plus_analysis import format_rouge


def split_sentences(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    bare_sents = re.findall(r'%s (.+?) %s' %
                            (sentence_start_tag, sentence_end_tag), article)
    return bare_sents


def count_data_by_suffix(path, suffix='json'):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.%s' % suffix)

    def match(name): return bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def read_json(fin, key='decode', k=3):
    js_data = json.loads(fin.read())
    abs_sents = js_data[key][:k]
    return abs_sents


def read_multi_line(fin):
    abs_sents = []
    for line in fin.readlines():
        abs_sents.append(make_html_safe(line.strip()))

    return abs_sents


def count_sentences(data):
    count_dict = defaultdict(int)
    for ele in data:
        for sent in ele:
            count_dict[sent] += 1
    count_sorted = sorted(count_dict.items(),
                          key=lambda x: x[1], reverse=True)
    return count_sorted


def write_to_outfile(data, output_path, num_samples=1000):
    with codecs.open(output_path, "w", encoding="utf-8") as f:
        for ele in data[:num_samples]:
            ele_str = "%s || %s\n" % (ele[0], ele[1])
            f.write(ele_str)


def get_co_sent():
    data_root_path = './output'
    preds_path = join(data_root_path, 'new_model_preds.txt')
    refs_path = join(data_root_path, 'test_refs.txt')

    count_root_path = '../../opennmt_summary/test/data'
    preds_count_path = join(count_root_path, 'new_model_preds_count.txt')
    refs_count_path = join(count_root_path, 'test_refs_count.txt')

    output_path = './output/data_analysis/co_sentences.txt'
    output_file = open(output_path, 'w')

    def load_data(data_path, is_count=False):
        res = []
        with open(data_path) as f:
            if is_count:
                for line in f.readlines():
                    res.append(line.strip().split("||")[0])
            else:
                for line in f.readlines():
                    res.append(line.strip())
        return res

    preds_data = load_data(preds_path, is_count=False)
    refs_data = load_data(refs_path, is_count=False)

    preds_count_data = load_data(preds_count_path, is_count=True)
    refs_count_data = load_data(refs_count_path, is_count=True)

    count_data_all = list(set(preds_count_data) & set(refs_count_data))

    # import pdb
    # pdb.set_trace()
    for query in tqdm(count_data_all):
        co_idx = []
        for i, (pred, ref) in enumerate(zip(preds_data, refs_data)):
            if query in pred and query in ref:
                co_idx.append(str(i + 1))

        if len(co_idx) > 0:
            idx_str = ", ".join(co_idx)
            output_file.write("%s || co_freq: %s || co_idxs: %s\n" %
                              (query, len(co_idx), idx_str))

    output_file.close()


def count_article_stats(args):
    input_path = join(args.in_path, args.split)
    suffix = args.suffix

    n_data = count_data_by_suffix(input_path, suffix=suffix)
    len_art = []
    len_abs = []

    len_sent_art = []
    len_sent_abs = []
    for i in tqdm(range(n_data)):
        with open(join(input_path, '{}.{}'.format(i, suffix))) as f:
            data = json.loads(f.read())
            art_sents = data['article']
            abs_sents = data['abstract']
            len_art.append(len(art_sents))
            len_abs.append(len(abs_sents))
            for ele in art_sents:
                len_sent_art.append(len(ele.split()))
            for ele in abs_sents:
                len_sent_abs.append(len(ele.split()))

    def print_data(data, name):
        print("%s: " % name)
        print("Total: %d, Min: %.2f, Max: %.2f, Mean: %.2f, Std: %.2f" %
              (len(data), np.min(data), np.max(data), np.mean(data), np.std(data)))
        percent = np.percentile(data, [50, 60, 70, 80, 90])
        print("Percentile: 50%%: %.2f, 60%%: %.2f, 70%%: %.2f, 80%%: %.2f, 90%%: %.2f\n" %
              tuple(percent))
    print_data(len_art, name='Length of article (Number of sentences)')
    print_data(len_abs, name='Length of abstract (Number of sentences)')
    print_data(
        len_sent_art, name='Length of sentences in article (Number of words)')
    print_data(
        len_sent_abs, name='Length of sentences in abstract (Number of words)')


def count_article_sent(args):
    input_path = args.in_path
    output_path = args.out_path
    suffix = args.suffix
    num_samples = args.n_samples

    n_data = count_data_by_suffix(input_path, suffix=suffix)
    # import pdb
    # pdb.set_trace()
    summaries = []
    for i in tqdm(range(n_data)):
        with open(join(input_path, '{}.{}'.format(i, suffix))) as f:
            data = json.loads(f.read())
            article_sents = list(
                map(lambda x: make_html_safe(x), data['article']))
            summaries.append(article_sents)

    count_sorted = count_sentences(summaries)
    write_to_outfile(count_sorted, output_path, num_samples=num_samples)


def merge_to_one_file(args):
    input_path = args.in_path
    output_path = args.out_path
    suffix = args.suffix

    n_data = count_data_by_suffix(input_path, suffix=suffix)
    # import pdb
    # pdb.set_trace()
    txt_file = codecs.open(output_path, "w", encoding='utf-8')
    for i in tqdm(range(n_data)):
        with open(join(input_path, '{}.{}'.format(i, suffix))) as f:
            # abs_sents = read_multi_line(f)
            abs_sents = read_json(f)
            abs_sents_tag = " ".join(
                map(lambda x: "<t> %s </t>" % x, abs_sents))
            txt_file.write(abs_sents_tag + "\n")

    txt_file.close()


def count_scores(args):
    input_path = join(args.in_path, args.split)
    output_path = args.out_path
    suffix = args.suffix

    n_data = count_data_by_suffix(input_path, suffix=suffix)
    txt_file = codecs.open(output_path, "w", encoding='utf-8')
    score_count_dict = defaultdict(list)
    # thresholds = [0.1 * i for i in range(1, 10)]
    thresholds = [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
                  (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for i in tqdm(range(n_data)):
        with open(join(input_path, '{}.{}'.format(i, suffix))) as f:
            data = json.loads(f.read())
            # score = data['score']
            score_dict = {
                # 'rouge_w_r': list(map(lambda x: x[0], data['rouge_w_r'])),
                'rouge_l_r': data['rouge_l_r'],
                # 'rouge_n1_r': list(map(lambda x: x[0], data['rouge_n1_r'])),
                # 'rouge_n2_r': list(map(lambda x: x[0], data['rouge_n2_r']))
                'rouge_l_r_max': data['rouge_l_r_max']
            }

            for key, score in score_dict.items():
                count_one = [0] * (len(thresholds) + 1)
                count_one[0] = len(score)
                for j, (low, high) in enumerate(thresholds):
                    for k, ele in enumerate(score):
                        if key == 'rouge_l_r_max':
                            data = ele
                            num_unit = 1
                        else:
                            arr = list(map(float, ele.split(", ")))
                            data = score_dict['rouge_l_r_max'][k]
                            num_unit = len(arr)
                        if low < data <= high:
                            count_one[j + 1] += num_unit

                score_count_dict[key].append(count_one)
                if key == 'rouge_l_r_max':
                    txt_file.write(", ".join(map(str, count_one)) + "\n")

    num_sum = 1
    for key, score_count in score_count_dict.items():
        for i in range(len(thresholds) + 1):
            data = list(map(lambda x: x[i], score_count))
            data = list(filter(lambda x: x > 0.1, data))
            num_count = sum(data)
            if i == 0:
                print("Total number of %s: %d (avg sent: %.2f)" %
                      (key, num_count, 1.0 * num_count / (len(score_count) + 1e-3)))
                num_sum = num_count
            else:
                if key == 'rouge_l_r_max':
                    rate_str = '(%.2f%%)' % (
                        100.0 * num_count / (num_sum + 1e-3))
                else:
                    num_total = sum(
                        map(lambda x: x[i], score_count_dict['rouge_l_r_max']))
                    rate_str = '(avg: %.2f)' % (
                        1.0 * num_count / (num_total + 1e-3))
                    percent = np.percentile(data, [50, 60, 70, 80, 90])
                    percent_str = "Percentile: 50%%: %.2f, 60%%: %.2f, 70%%: %.2f, 80%%: %.2f, 90%%: %.2f" % tuple(
                        percent)
                print("Number of %s (%.1f-%.1f): %d %s" %
                      (key, thresholds[i - 1][0], thresholds[i - 1][1], num_count, rate_str))
                if key != 'rouge_l_r_max':
                    print("%s: %s" % (key, percent_str))

    txt_file.close()


def switch_rewrite_helper(js_data, threshold, is_float, split, threshold_str):
    tokenize = compose(list, _split_words)
    ext_idxs = js_data['extract_preds']
    ext_sents_raw = [js_data['article'][idx] for idx in ext_idxs]
    ext_sents = tokenize(ext_sents_raw)
    dec_sents_raw = js_data['decode']
    dec_sents = tokenize(dec_sents_raw)
    abs_sents = tokenize(js_data['abstract'])

    num_switch = 0
    res_sents = []
    use_ext = []
    res_score = []
    if not is_float and 'ext_gt' == threshold:
        if split == 'val':
            ext_idxs_gt = js_data['extracted']
            res_sents = [js_data['article'][idx] for idx in ext_idxs_gt]
            use_ext = [True] * len(ext_idxs_gt)
            res_score = js_data['score']
        else:
            res_sents = ext_sents_raw
            use_ext = [True] * len(ext_idxs)

    else:
        rouge_ext_all = []
        rouge_dec_all = []
        for abst in abs_sents:
            rouge_ext = list(map(compute_rouge_l(reference=abst, mode='f'),
                                 ext_sents))
            rouge_ext_all.append(rouge_ext)
            if not is_float:
                rouge_dec = list(map(compute_rouge_l(reference=abst, mode='f'),
                                     dec_sents))
                rouge_dec_all.append(rouge_dec)

        for idx in range(len(ext_idxs)):
            rouge_ext_max = max(map(lambda x: x[idx], rouge_ext_all))
            rouge_dec_max = rouge_ext_max
            if is_float:
                compared = threshold
            elif 'ext_pred' == threshold:
                compared = -1.0
            else:
                rouge_dec_max = max(map(lambda x: x[idx], rouge_dec_all))
                compared = rouge_dec_max

            if rouge_ext_max > compared:
                res_sents.append(ext_sents_raw[idx])
                res_score.append(rouge_ext_max)
                use_ext.append(True)
                num_switch += 1
            else:
                res_sents.append(dec_sents_raw[idx])
                res_score.append(rouge_dec_max)
                use_ext.append(False)

    out_data = {}
    out_data['id'] = js_data['id']
    out_data['article'] = js_data['article']
    out_data['abstract'] = js_data['abstract']
    out_data['decode'] = res_sents
    if split == 'val':
        out_data['extracted'] = js_data['extracted']
    out_data['extract_preds'] = ext_idxs
    # calculating rouge score
    rouge_str = format_rouge(
        [res_sents], [out_data['abstract']], split=split, threshold=threshold_str)
    out_data['rouge'] = rouge_str
    out_data['score'] = res_score
    out_data['use_ext'] = use_ext
    out_data['is_top'] = js_data['is_top']

    return out_data, num_switch


def switch_rewrite(args):
    input_path = join(args.in_path, args.split)
    output_path = "%s_%s" % (args.out_path, args.threshold)
    suffix = args.suffix
    is_float = False
    try:
        threshold = float(args.threshold)
        is_float = True
    except ValueError:
        threshold = args.threshold
    split = args.split

    if not exists(output_path):
        os.mkdir(output_path)
    output_data_path = join(output_path, split)
    if not exists(output_data_path):
        os.mkdir(output_data_path)

    n_data = count_data_by_suffix(input_path, suffix=suffix)
    num_count = 0
    num_switch = 0
    for i in tqdm(range(n_data)):
        with open(join(input_path, '{}.{}'.format(i, suffix))) as fin:
            js_data = json.loads(fin.read())
            num_count += len(js_data['extract_preds'])
            out_data, n_swi = switch_rewrite_helper(
                js_data, threshold, is_float, split, args.threshold)
            num_switch += n_swi
            with open(join(output_data_path, '{}.json'.format(i)),
                      'w') as fout:
                json.dump(out_data, fout, indent=4)
            # import pdb
            # pdb.set_trace()

    print("%s:%s Total: %d, Number of switch: %d (%.2f%%)" %
          (split.upper(), args.threshold.upper(), num_count, num_switch, 100.0 * num_switch / (num_count + 1e-3)))


def extract_span_analysis(args):
    input_path = join(args.in_path, args.split)
    output_path = args.out_path
    suffix = args.suffix
    split = args.split

    if not exists(output_path):
        os.mkdir(output_path)
    output_data_path = join(output_path, split)
    if not exists(output_data_path):
        os.mkdir(output_data_path)

    n_data = count_data_by_suffix(input_path, suffix=suffix)
    thresholds = [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
                  (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    txt_file_all = [0] * len(thresholds)
    out_data_all = []
    for i, (low, high) in enumerate(thresholds):
        txt_path = join(output_data_path, 'span_%.1f_%.1f' % (low, high))
        txt_file_all[i] = codecs.open(txt_path, "w", encoding='utf-8')
        out_data_all.append([])

    metric_name = 'rouge_w_r'
    for i in tqdm(range(n_data)):
        with open(join(input_path, '{}.{}'.format(i, suffix))) as f:
            js_data = json.loads(f.read())
            score = list(map(lambda x: x[0], js_data[metric_name]))
            ext_idxs_gt = list(map(lambda x: x[0], js_data['extracted']))
            art_sents = [js_data['article'][idx] for idx in ext_idxs_gt]
            abs_sents = js_data['abstract']
            for j, (s, arts, abst) in enumerate(zip(score, art_sents, abs_sents)):
                find_idx = -1
                for k, (low, high) in enumerate(thresholds):
                    if low < s <= high:
                        find_idx = k
                        break
                out_data = {
                    'article_idx': i,
                    metric_name: '%.2f' % (100 * s),
                    'art_sent': arts,
                    'abs_sent': abst
                }
                out_data_all[k].append(out_data)

    for i, out_data in enumerate(out_data_all):
        out_data = sorted(out_data, key=lambda x: float(x[metric_name]))
        for ele in out_data:
            out_str = json.dumps(ele, indent=4)
            txt_file_all[i].write(out_str + '\n')

    for txt_file in txt_file_all:
        txt_file.close()


def multi_sents_analysis(args):
    input_path = join(args.in_path, args.split)
    output_path = args.out_path
    suffix = args.suffix
    split = args.split

    n_data = count_data_by_suffix(input_path, suffix=suffix)
    txt_file = codecs.open(output_path, "w", encoding='utf-8')
    n_repeat = 0
    n_repeat_max = 0

    def is_max_repeat(arr):
        arr = list(map(lambda x: x[0], arr))
        arr_uniq = np.unique(arr)
        if len(arr_uniq) < len(arr):
            return True
        else:
            return False

    n_total_sent = 0
    for i in tqdm(range(n_data)):
        with open(join(input_path, '{}.{}'.format(i, suffix))) as fin:
            js_data = json.loads(fin.read())
            art_sents = js_data['article']
            ext_idx = list(map(lambda x: list(
                map(int, x.split(", "))), js_data['extracted']))
            if is_max_repeat(ext_idx):
                n_repeat_max += 1
            ext_idx_concat = list(concat(ext_idx))
            ext_idx_uniq = np.unique(ext_idx_concat)
            ext_idx_max = list(map(lambda x: x[0], ext_idx))
            if len(ext_idx_uniq) < len(ext_idx_concat):
                n_repeat += 1

            abs_sents = [make_html_safe(art_sents[i]) for i in ext_idx_max]
            # import pdb
            # pdb.set_trace()
            abs_sents_tag = " ".join(
                map(lambda x: "<t> %s </t>" % x, abs_sents))
            n_total_sent += len(ext_idx_max)
            txt_file.write(abs_sents_tag + "\n")

    txt_file.close()
    print("Total: %d, Number of repeat: %d (%.2f%%), Number of max repeat: %d (%.2f%%)" % (
        n_data, n_repeat, 100.0 * (n_repeat / (n_data + 1e-3)), n_repeat_max, 100.0 * (n_repeat_max / (n_data + 1e-3))))
    print("Total sentences: %d (avg: %.2f)" %
          (n_total_sent, n_total_sent / (n_data + 1e-3)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-func', '--function', required=True,
                        help='function to be selected.')
    parser.add_argument('-in', '--in_path', required=True,
                        help='Path to input path.')
    parser.add_argument('-out', '--out_path', required=False, default='./output/data_analysis',
                        help='Path to output file.')
    parser.add_argument('-suf', '--suffix', required=False, default='json',
                        help='Suffix of file.')
    parser.add_argument('-n_sam', '--n_samples', required=False, type=int, default=1000,
                        help='Number of samples for output file.')
    parser.add_argument('-thre', '--threshold', required=False, type=str, default='0.5',
                        help='threshold for switch')
    parser.add_argument('-spl', '--split', required=False, type=str, default='test',
                        help='Test or Validation.')
    args = parser.parse_args()

    function = args.function
    if function == 'count_art_stats':
        count_article_stats(args)
    elif function == 'count_art_sent':
        count_article_sent(args)
    elif function == 'merge_to_one_file':
        merge_to_one_file(args)
    elif function == 'count_scores':
        count_scores(args)
    elif function == 'switch_rewrite':
        switch_rewrite(args)
    elif function == 'extract_span':
        extract_span_analysis(args)
    elif function == 'multi_sents':
        multi_sents_analysis(args)
    else:
        print('Unkown function name, please input again!!!')


if __name__ == '__main__':
    main()
