""" Evaluate the baselines ont ROUGE/METEOR"""
import argparse
import json
import os
from os.path import join, exists
from time import time
from datetime import timedelta
from evaluate import eval_meteor, eval_rouge
from pyrouge import Rouge155


# try:
#     _DATA_DIR = os.environ['DATA']
# except KeyError:
#     print('please use environment variable to specify data directories')
# DATA_DIR = '/home/zhangwj/code/nlp/summarization/dataset/raw/CNN_Daily/fast_abs_rl/finished_files'


def print_official_rouge(results_dict):
    for idx in ['1', '2', 'l']:
        head_prefix = "ROUGE-%s" % idx.upper()
        val_prefix = "rouge_%s" % idx
        res = ">> {}-R: {:.2f}, {}-P: {:.2f}, {}-F: {:.2f}".format(
            head_prefix, results_dict["%s_recall" % val_prefix] * 100,
            head_prefix, results_dict["%s_precision" % val_prefix] * 100,
            head_prefix, results_dict["%s_f_score" % val_prefix] * 100)
        print(res)
    return res


def main(args):
    # dec_dir = join(args.decode_dir, 'output')
    # with open(join(args.decode_dir, 'log.json')) as f:
    #     split = json.loads(f.read())['split']
    # ref_dir = join(_DATA_DIR, 'refs', split)
    # assert exists(ref_dir)
    start = time()
    dec_dir = args.decode_dir
    ref_dir = args.refer_dir
    if args.rouge:
        dec_pattern = r'(\d+).dec'
        ref_pattern = '#ID#.ref'
        output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir)
        metric = 'rouge'
    else:
        dec_pattern = '[0-9]+.dec'
        ref_pattern = '[0-9]+.ref'
        output = eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir)
        metric = 'meteor'
    print(output)
    # with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
    #     f.write(output)
    r = Rouge155()
    res_dict = r.output_to_dict(output)
    print_official_rouge(res_dict)
    print('finished in {}'.format(timedelta(seconds=time() - start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the output files for the RL full models')

    # choose metric to evaluate
    metric_opt = parser.add_mutually_exclusive_group(required=True)
    metric_opt.add_argument('--rouge', action='store_true',
                            help='ROUGE evaluation')
    metric_opt.add_argument('--meteor', action='store_true',
                            help='METEOR evaluation')

    parser.add_argument('--decode_dir', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('--refer_dir', action='store', required=True,
                        help='directory of references')

    args = parser.parse_args()
    main(args)
