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

from cytoolz import curry, compose
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


def read_json(fin, key='decode'):
    js_data = json.loads(fin.read())
    abs_sents = js_data[key]
    return abs_sents


def read_multi_line(fin):
    abs_sents = []
    for line in fin.readlines():
        abs_sents.append(make_html_safe(line.strip()))

    return abs_sents


def merge_to_one_file():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--in_path', required=True,
                        help='Path to input path.')
    parser.add_argument('-out', '--out_path', required=True,
                        help='Path to output file.')
    parser.add_argument('-suf', '--suffix', required=True,
                        help='Suffix of file.')
    args = parser.parse_args()
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


def count_article_sent():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--in_path', required=True,
                        help='Path to input path.')
    parser.add_argument('-out', '--out_path', required=True,
                        help='Path to output file.')
    parser.add_argument('-suf', '--suffix', required=True,
                        help='Suffix of file.')
    parser.add_argument('-n_sam', '--n_samples', required=False, type=int, default=1000,
                        help='Number of samples for output file.')
    args = parser.parse_args()
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


def count_scores():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--in_path', required=True,
                        help='Path to input path.')
    parser.add_argument('-out', '--out_path', required=True,
                        help='Path to output file.')
    parser.add_argument('-suf', '--suffix', required=True,
                        help='Suffix of file.')

    args = parser.parse_args()
    input_path = args.in_path
    output_path = args.out_path
    suffix = args.suffix

    n_data = count_data_by_suffix(input_path, suffix=suffix)

    txt_file = codecs.open(output_path, "w", encoding='utf-8')
    score_count = []
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for i in tqdm(range(n_data)):
        with open(join(input_path, '{}.{}'.format(i, suffix))) as f:
            data = json.loads(f.read())
            score = data['score']
            count_one = [0] * (len(thresholds) + 1)
            count_one[0] = len(score)
            for ele in score:
                if ele > thresholds[4]:
                    count_one[5] += 1
                if ele > thresholds[3]:
                    count_one[4] += 1
                if ele > thresholds[2]:
                    count_one[3] += 1
                if ele > thresholds[1]:
                    count_one[2] += 1
                if ele > thresholds[0]:
                    count_one[1] += 1

            score_count.append(count_one)
            txt_file.write(", ".join(map(str, count_one)) + "\n")

    num_sum = 1
    for i in range(len(thresholds) + 1):
        num_count = sum(map(lambda x: x[i], score_count))
        if i == 0:
            print("Total number of score: %d (avg sent: %.2f)" %
                  (num_count, 1.0 * num_count / (len(score_count) + 1e-3)))
            num_sum = num_count
        else:
            print("Number of score with threshold %.2f: %d (%.2f%%)" %
                  (thresholds[i - 1], num_count, 100.0 * num_count / (num_sum + 1e-3)))

    txt_file.close()


def switch_rewrite():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--in_path', required=True,
                        help='Path to input path.')
    parser.add_argument('-out', '--out_path', required=True,
                        help='Path to output file.')
    parser.add_argument('-suf', '--suffix', required=True,
                        help='Suffix of file.')
    parser.add_argument('-thre', '--threshold', required=False, type=str, default='0.5',
                        help='threshold for switch')
    parser.add_argument('-spl', '--split', required=False, type=str, default='test',
                        help='Test or Validation.')
    args = parser.parse_args()

    input_path = join(args.in_path, args.split)
    output_path = "%s_%s" % (args.out_path, args.threshold)
    suffix = args.suffix
    try:
        threshold = float(args.threshold)
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
            tokenize = compose(list, _split_words)
            ext_idxs = js_data['extract_preds']
            ext_sents_raw = [js_data['article'][idx] for idx in ext_idxs]
            ext_sents = tokenize(ext_sents_raw)
            dec_sents_raw = js_data['decode']
            dec_sents = tokenize(dec_sents_raw)
            abs_sents = tokenize(js_data['abstract'])
            rouge_ext_all = []
            rouge_dec_all = []
            for abst in abs_sents:
                rouge_ext = list(map(compute_rouge_l(reference=abst, mode='f'),
                                     ext_sents))
                rouge_dec = list(map(compute_rouge_l(reference=abst, mode='f'),
                                     dec_sents))
                rouge_ext_all.append(rouge_ext)
                rouge_dec_all.append(rouge_dec)

            res_sents = []
            use_ext = []
            res_score = []
            num_count += len(ext_idxs)
            for idx in range(len(ext_idxs)):
                rouge_ext_max = max(map(lambda x: x[idx], rouge_ext_all))
                rouge_dec_max = max(map(lambda x: x[idx], rouge_dec_all))

                if rouge_ext_max > rouge_dec_max:
                    res_sents.append(ext_sents_raw[idx])
                    use_ext.append(True)
                    num_switch += 1
                else:
                    res_sents.append(dec_sents_raw[idx])
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
                [res_sents], [out_data['abstract']], split=split, threshold=args.threshold)
            out_data['rouge'] = rouge_str
            out_data['score'] = res_score
            out_data['use_ext'] = use_ext
            out_data['is_top'] = js_data['is_top']

            with open(join(output_data_path, '{}.json'.format(i)),
                      'w') as fout:
                json.dump(out_data, fout, indent=4)
            # import pdb
            # pdb.set_trace()

    print("%s:%s Total: %d, Number of switch: %d (%.2f%%)" %
          (split.upper(), args.threshold, num_count, num_switch, 100.0 * num_switch / (num_count + 1e-3)))


def main():
    # fire.Fire()
    # count_article_sent()
    merge_to_one_file()
    # count_scores()
    # switch_rewrite()


if __name__ == '__main__':
    main()
