""" make reference text files needed for ROUGE evaluation """
import argparse
import codecs
import json
import os
from os.path import join, exists
from time import time
from datetime import timedelta
from tqdm import tqdm
from utils import count_data
from decoding import make_html_safe

# try:
#     DATA_DIR = os.environ['DATA']
# except KeyError:
#     print('please use environment variable to specify data directories')
DATA_DIR = '/home/zhangwj/code/nlp/summarization/dataset/raw/CNN_Daily/fast_abs_rl/finished_files'


def dump(split):
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    dump_dir = join(DATA_DIR, 'refs', split)
    n_data = count_data(data_dir)
    for i in range(n_data):
        print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100 * i / n_data),
              end='')
        with open(join(data_dir, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        abs_sents = data['abstract']
        with open(join(dump_dir, '{}.ref'.format(i)), 'w') as f:
            f.write(make_html_safe('\n'.join(abs_sents)))
    print('finished in {}'.format(timedelta(seconds=time() - start)))


def get_lead_3_and_refs(split, output_path):
    start = time()
    # print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    lead_3_path = join(output_path, "lead_3_baseline.txt")
    refs_path = join(output_path, "references.txt")

    lead_3_file = codecs.open(lead_3_path, "w", encoding='utf-8')
    refs_file = codecs.open(refs_path, "w", encoding='utf-8')
    n_data = count_data(data_dir)
    for i in tqdm(range(n_data)):
        with open(join(data_dir, '{}.json'.format(i))) as f:
            data = json.loads(f.read())

        # lead_3
        article_sents = data['article'][:3]
        abs_sents = data['abstract']

        article_sents_tag = " ".join(
            map(lambda x: "<t> %s </t>" % make_html_safe(x), article_sents))
        abs_sents_tag = " ".join(map(lambda x: "<t> %s </t>" %
                                     make_html_safe(x), abs_sents))
        lead_3_file.write(article_sents_tag + "\n")
        refs_file.write(abs_sents_tag + "\n")

    lead_3_file.close()
    refs_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-out', '--out_path', required=True,
                        help='Path to output.')
    args = parser.parse_args()
    # for split in ['val', 'test']:  # evaluation of train data takes too long
    #     if not exists(join(DATA_DIR, 'refs', split)):
    #         os.makedirs(join(DATA_DIR, 'refs', split))
    #     dump(split)
    split = 'test'
    output_path = args.out_path
    get_lead_3_and_refs(split, output_path)


if __name__ == '__main__':
    main()
