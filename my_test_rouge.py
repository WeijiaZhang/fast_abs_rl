# -*- encoding: utf-8 -*-
import argparse
import re
import os
import time
import shutil
import sys
import codecs

import subprocess as sp
import logging
import tempfile
import pyrouge

from evaluate import eval_rouge_by_cmd


def split_sentences_no_tag(article):
    article = article.strip()
    good_sents = []
    while len(article) > 0:
        try:
            end_sent_idx = article.index(' .')
        except ValueError:
            end_sent_idx = len(article)
        sent = article[:end_sent_idx + 2]
        if len(sent) > 3:
            good_sents.append(sent)
        article = article[end_sent_idx + 3:]
    # import pdb
    # pdb.set_trace()
    return good_sents


def split_sentences(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    bare_sents = re.findall(r'%s (.+?) %s' %
                            (sentence_start_tag, sentence_end_tag), article)
    return bare_sents

# convenient decorator


def register_to_registry(registry):
    def _register(func):
        registry[func.__name__] = func
        return func
    return _register


baseline_registry = {}
register = register_to_registry(baseline_registry)

# baseline methods


@register
def all_sentences(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents


@register
def first_sentence(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    ''' use sentence tags to output the first sentence of an article as its summary. '''
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents[:1]


@register
def first_two_sentences(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents[:2]


@register
def first_three_sentences(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    sents = split_sentences_no_tag(article)
    return sents[:3]


@register
def verbatim(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents


@register
def pre_sent_tag_verbatim(article):
    sents = article.split('<t>')
    good_sents = []
    for sent in sents:
        sent = sent.strip()
        if len(sent.split()) > 0:
            good_sents.append(sent)
    # print(good_sents)
    return good_sents


@register
def sent_tag_verbatim(article):
    sents = split_sentences(article, '<t>', '</t>')
    # print(sents)
    return sents


@register
def sent_tag_p_verbatim(article):
    bare_article = article.strip()
    bare_article += ' </t>'
    sents = split_sentences(bare_article, '<t>', '</t>')
    # print(sents)
    return sents


@register
def adhoc_old0(article):
    sents = split_sentences(article, '<t>', '</t>')
    good_sents = []
    for sent in sents:
        # Remove <unk>
        tokens = [x for x in sent.split() if x != '<unk>']
        # Ignore length 1 sententces
        if len(tokens) > 1:
            good_sents.append(' '.join(tokens))
    return good_sents


@register
def full(article):
    return [article]


@register
def adhoc_base(article):
    article += ' </t> </t>'
    first_end = article.index(' </t> </t>')
    article = article[:first_end] + ' </t>'
    sents = split_sentences(article)
    good_sents = []
    for sent in sents:
        # Remove <unk>
        tokens = [x for x in sent.split() if x != '<unk>']
        # Ignore length 1 sententces
        if len(tokens) > 1:
            good_sents.append(' '.join(tokens))
    return good_sents


@register
def no_sent_tag(article):
    good_sents = split_sentences_no_tag(article)
    return good_sents


@register
def no_sent_tag_old(article):
    article = article.strip()
    try:
        if article[-1] != '.':
            article += ' .'
    except:
        article += ' .'
    good_sents = list(re.findall(r'.+?\.', article))
    return good_sents


@register
def second_sentence(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents[1:2]


def test_rouge(summaries, references, rouge_args=[]):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = ".rouge-tmp-{}".format(current_time)
    system_dir = os.path.join(tmp_dir, 'system')
    model_dir = os.path.join(tmp_dir, 'model')
    args_str = ' '.join(map(str, rouge_args))
    try:
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
            os.mkdir(system_dir)
            os.mkdir(model_dir)

        candidates = [" ".join(ele) for ele in summaries]
        references = [" ".join(ele[0]) for ele in references]
        assert len(candidates) == len(references)
        cnt = len(candidates)
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(os.path.join(system_dir, "cand.%i.txt" % i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(os.path.join(model_dir, "ref.%i.txt" % i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155()
        r.system_dir = system_dir
        r.model_dir = model_dir
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        results_dict = r.output_to_dict(rouge_results)
        return results_dict
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)


# def write_sentences_old(summaries, references, system_dir, model_dir):
#     for i, (summary, candidates) in enumerate(zip(summaries, references)):
#         summary_file = '%i.txt' % i
#         for j, candidate in enumerate(candidates):
#             candidate_file = '%i.%i.txt' % (i, j)
#             with open(os.path.join(model_dir, candidate_file), 'w', encoding="utf-8") as f:
#                 f.write('\n'.join(candidate))

#         with open(os.path.join(system_dir, summary_file), 'w', encoding="utf-8") as f:
#             f.write('\n'.join(summary))


# def write_sentences(summaries, references, system_dir, model_dir):
#     for i, (summary, candidate) in enumerate(zip(summaries, references)):
#         if i > 100:
#             break
#         summary_file = '%i.txt' % i
#         candidate_file = '%i.txt' % i
#         with open(os.path.join(model_dir, candidate_file), 'w', encoding="utf-8") as f:
#             f.write('\n'.join(candidate))

#         with open(os.path.join(system_dir, summary_file), 'w', encoding="utf-8") as f:
#             f.write('\n'.join(summary))


# def eval_rouge_cmd_helper(dec_pattern, dec_dir, ref_pattern, ref_dir,
#                           cmd='-c 95 -r 1000 -n 2 -m', system_id=1):
#     """ evaluate by original Perl implementation"""
#     # silence pyrouge logging
#     _ROUGE_PATH = os.environ['ROUGE']
#     log.get_global_console_logger().setLevel(logging.WARNING)
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         Rouge155.convert_summaries_to_rouge_format(
#             dec_dir, os.path.join(tmp_dir, 'dec'))
#         Rouge155.convert_summaries_to_rouge_format(
#             ref_dir, os.path.join(tmp_dir, 'ref'))
#         Rouge155.write_config_static(
#             os.path.join(tmp_dir, 'dec'), dec_pattern,
#             os.path.join(tmp_dir, 'ref'), ref_pattern,
#             os.path.join(tmp_dir, 'settings.xml'), system_id
#         )
#         cmd = (os.path.join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
#                + ' -e {} '.format(os.path.join(_ROUGE_PATH, 'data'))
#                + cmd
#                + ' -a {}'.format(os.path.join(tmp_dir, 'settings.xml')))
#         output = sp.check_output(cmd.split(' '), universal_newlines=True)
#     return output


# def eval_rouge_by_cmd(summaries, references):
#     tmp_root_dir = "./rouge_tmp"
#     current_time = "rouge_{}".format(time.strftime(
#         '%Y-%m-%d-%H-%M-%S', time.localtime()))
#     tmp_dir = os.path.join(tmp_root_dir, )
#     system_filename_pattern = '(\d+).txt'
#     model_filename_pattern = '#ID#.txt'
#     system_dir = os.path.join(tmp_dir, 'system')
#     model_dir = os.path.join(tmp_dir, 'model')
#     try:
#         if not os.path.isdir(tmp_dir):
#             os.mkdir(tmp_dir)
#             os.mkdir(system_dir)
#             os.mkdir(model_dir)
#         assert len(summaries) == len(references)
#         write_sentences(summaries, references, system_dir, model_dir)

#         output = eval_rouge_cmd_helper(system_filename_pattern, system_dir,
#                                        model_filename_pattern, model_dir)

#         print(output)
#         r = Rouge155()
#         res_dict = r.output_to_dict(output)
#         return res_dict
#     finally:
#         # pass
#         if os.path.isdir(tmp_dir):
#             shutil.rmtree(tmp_dir)


def eval_rouge_by_sentence(summaries, references, rouge_args=[]):
    '''
    Args:
        summaries: [[sentence]]. Each summary is a list of strings (sentences)
        references: [[[sentence]]]. Each reference is a list of candidate summaries.
        rouge_args: [string]. A list of arguments to pass to the ROUGE CLI.
    '''
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = ".rouge-tmp-{}".format(current_time)
    system_dir = os.path.join(tmp_dir, 'system')
    model_dir = os.path.join(tmp_dir, 'model')
    args_str = ' '.join(map(str, rouge_args))
    try:
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
            os.mkdir(system_dir)
            os.mkdir(model_dir)
        assert len(summaries) == len(references)
        write_sentences(summaries, references, system_dir, model_dir)
        r = pyrouge.Rouge155()
        r.system_dir = system_dir
        r.model_dir = model_dir
        # r.system_filename_pattern = '(\d+).txt'
        # r.model_filename_pattern = '#ID#.\d+.txt'
        r.system_filename_pattern = '(\d+).txt'
        r.model_filename_pattern = '#ID#.txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
        return results_dict
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)


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


def print_rouge_scores(scores):
    # g_headers = ['rouge_1/r_score', 'rouge_1/p_score', 'rouge_1/f_score', 'rouge_2/r_score',
    #              'rouge_2/p_score', 'rouge_2/f_score', 'rouge_l/r_score', 'rouge_l/p_score', 'rouge_l/f_score']
    # print(g_headers)
    for idx in ['1', '2', 'l']:
        head_prefix = "ROUGE-%s" % idx.upper()
        val_prefix = "rouge_%s" % idx
        res = ">> {}-R: {:.2f}, {}-P: {:.2f}, {}-F: {:.2f}".format(
            head_prefix, scores["%s/r_score" % val_prefix] * 100,
            head_prefix, scores["%s/p_score" % val_prefix] * 100,
            head_prefix, scores["%s/f_score" % val_prefix] * 100)
        print(res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required=True,
                        help='Path to the tokenized source file. One sample per line with sentence tags.')
    parser.add_argument('-t', '--target', required=True,
                        help='Path to the tokenized target file. One sample per line with sentence tags.')
    parser.add_argument('-ms', '--method-src', required=False, default='all_sentences',
                        choices=baseline_registry.keys(), help='Baseline method for source to use.')
    parser.add_argument('-mt', '--method-tgt', required=False, default='all_sentences',
                        choices=baseline_registry.keys(), help='Baseline method for target to use.')
    parser.add_argument('-thre', '--threshold', required=False, type=str, default='0.5',
                        help='threshold for switch')
    parser.add_argument('-spl', '--split', required=False, type=str, default='test',
                        help='Test or Validation.')
    parser.add_argument('--no-stemming', dest='stemming',
                        action='store_false', help='Turn off stemming in ROUGE.')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='The number of bootstrap samples used in ROUGE.')
    parser.add_argument('-g', '--google', dest='run_google_rouge', action='store_true',
                        help='Evaluate with the ROUGE implementation from google/seq2seq.')
    parser.add_argument('-p', '--pltrdy', dest='run_pltrdy_rouge', action='store_true',
                        help='Evaluate with the ROUGE implementation from git:pltrdy/rouge')
    args = parser.parse_args()

    process_src = baseline_registry[args.method_src]
    process_tgt = baseline_registry[args.method_tgt]
    threshold = args.threshold
    split = args.split

    source_file = codecs.open(args.source, encoding="utf-8")
    target_file = codecs.open(args.target, encoding="utf-8")

    # Read and preprocess generated summary
    n_source = 0
    references = []
    summaries = []
    for i, article in enumerate(source_file):
        summary = process_src(article)
        # import pdb
        # pdb.set_trace()
        summaries.append(summary)
        n_source += 1

    n_target = 0
    for i, article in enumerate(target_file):
        candidate = process_tgt(article)
        references.append(candidate)
        n_target += 1

    source_file.close()
    target_file.close()
    assert n_source == n_target, 'Source and target must have the same number of samples.'
    rouge_args = [
        '-c', 95,  # 95% confidence intervals, necessary for the dictionary conversion routine
        '-n', 2,  # up to bigram
        '-a',
        '-r', args.n_bootstrap,  # the number of bootstrap samples for confidence bounds
    ]
    if args.stemming:
        # add the stemming flag
        rouge_args += ['-m']

    t0 = time.time()
    if args.run_pltrdy_rouge:
        print(">> Performing pltrdy rouge")
        # p_scores = rouge_score.cal_rouge(summaries, references)
        # print_rouge_scores(p_scores)
    elif args.run_google_rouge:
        print(">> Performing google rouge")
        # g_scores = rouge(summaries, references)
        # print_rouge_scores(g_scores)
    else:
        print(">> Performing official ROUGE evaluation")
        # evaluate with official ROUGE script v1.5.5
        # results_dict = test_rouge(summaries, references, rouge_args=rouge_args)
        # results_dict = eval_rouge_by_sentence(
        #     summaries, references, rouge_args=rouge_args)

        # print('>> method_src: %s, method_tgt: %s' %
        #             (args.method_src, args.method_tgt))
        # print_official_rouge(results_dict)

        # saving lead_3 baseline data
        # def save_lead_3(summaries):
        #     with codecs.open("data/lead_3_baseline.txt.tgt", "w", encoding="utf-8") as fout:
        #         for summary in summaries:
        #             sent_tag = map(lambda x: "<t> %s </t>" % x, summary)
        #             fout.write(" ".join(sent_tag) + "\n")
        # save_lead_3(summaries)

        r_scores = eval_rouge_by_cmd(
            summaries, references, split=split, threshold=threshold, is_print_avg=True)
        print_official_rouge(r_scores)

    dt = time.time() - t0
    print('>> evaluated {} samples, took {:.3f}s, averaging {:.3f}s/sample'.format(
        n_target, dt, dt / (n_target + 1e-3)))


if __name__ == "__main__":
    main()
