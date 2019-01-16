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
            abs_sents = []
            for line in f.readlines():
                abs_sents.append(make_html_safe(line.strip()))

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


def main():
    pass


if __name__ == '__main__':
    # fire.Fire()
    # count_article_sent()
    merge_to_one_file()
