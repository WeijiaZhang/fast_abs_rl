""" evaluation scripts"""
import re
import os
from os.path import join, exists
import logging
import tempfile
import shutil
import subprocess as sp

from cytoolz import curry
import pyrouge
from pyrouge import Rouge155
from pyrouge.utils import log


try:
    _ROUGE_PATH = os.environ['ROUGE']
except KeyError:
    print('Warning: ROUGE is not configured')
    _ROUGE_PATH = None


def write_sentences(summaries, references, system_dir, model_dir):
    for i, (summary, candidate) in enumerate(zip(summaries, references)):
        # if i > 100:
        #     break
        summary_file = '%i.txt' % i
        candidate_file = '%i.txt' % i
        with open(os.path.join(model_dir, candidate_file), 'w', encoding="utf-8") as f:
            f.write('\n'.join(candidate))

        with open(os.path.join(system_dir, summary_file), 'w', encoding="utf-8") as f:
            f.write('\n'.join(summary))


def eval_rouge_cmd_helper(dec_pattern, dec_dir, ref_pattern, ref_dir,
                          cmd='-c 95 -r 1000 -n 2 -m', system_id=1):
    """ evaluate by original Perl implementation"""
    # silence pyrouge logging
    log.get_global_console_logger().setLevel(logging.WARNING)
    with tempfile.TemporaryDirectory() as tmp_dir:
        Rouge155.convert_summaries_to_rouge_format(
            dec_dir, os.path.join(tmp_dir, 'dec'))
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, os.path.join(tmp_dir, 'ref'))
        Rouge155.write_config_static(
            os.path.join(tmp_dir, 'dec'), dec_pattern,
            os.path.join(tmp_dir, 'ref'), ref_pattern,
            os.path.join(tmp_dir, 'settings.xml'), system_id
        )
        cmd = (os.path.join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
               + ' -e {} '.format(os.path.join(_ROUGE_PATH, 'data'))
               + cmd
               + ' -a {}'.format(os.path.join(tmp_dir, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output


def eval_rouge_by_cmd(summaries, references, split='test', threshold='0.5', is_print_avg=False):
    tmp_root_path = './rouge_tmp'
    tmp_dir = join(tmp_root_path, "%s_%s" % (split, threshold))
    system_filename_pattern = '(\d+).txt'
    model_filename_pattern = '#ID#.txt'
    system_dir = os.path.join(tmp_dir, 'system')
    model_dir = os.path.join(tmp_dir, 'model')
    try:
        if not exists(tmp_dir):
            os.mkdir(tmp_dir)
            os.mkdir(system_dir)
            os.mkdir(model_dir)
        assert len(summaries) == len(references)
        write_sentences(summaries, references, system_dir, model_dir)

        output = eval_rouge_cmd_helper(system_filename_pattern, system_dir,
                                       model_filename_pattern, model_dir)
        if is_print_avg:
            print(output)
        r = Rouge155()
        res_dict = r.output_to_dict(output)
        return res_dict
    finally:
        # pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)


def eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir,
               cmd='-c 95 -r 1000 -n 2 -m', system_id=1):
    """ evaluate by original Perl implementation"""
    # silence pyrouge logging
    assert _ROUGE_PATH is not None
    log.get_global_console_logger().setLevel(logging.WARNING)
    with tempfile.TemporaryDirectory() as tmp_dir:
        Rouge155.convert_summaries_to_rouge_format(
            dec_dir, join(tmp_dir, 'dec'))
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, join(tmp_dir, 'ref'))
        Rouge155.write_config_static(
            join(tmp_dir, 'dec'), dec_pattern,
            join(tmp_dir, 'ref'), ref_pattern,
            join(tmp_dir, 'settings.xml'), system_id
        )
        cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
               + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
               + cmd
               + ' -a {}'.format(join(tmp_dir, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output


try:
    _METEOR_PATH = os.environ['METEOR']
except KeyError:
    print('Warning: METEOR is not configured')
    _METEOR_PATH = None


def eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir):
    """ METEOR evaluation"""
    assert _METEOR_PATH is not None
    ref_matcher = re.compile(ref_pattern)
    refs = sorted([r for r in os.listdir(ref_dir) if ref_matcher.match(r)],
                  key=lambda name: int(name.split('.')[0]))
    dec_matcher = re.compile(dec_pattern)
    decs = sorted([d for d in os.listdir(dec_dir) if dec_matcher.match(d)],
                  key=lambda name: int(name.split('.')[0]))

    @curry
    def read_file(file_dir, file_name):
        with open(join(file_dir, file_name)) as f:
            return ' '.join(f.read().split())
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(join(tmp_dir, 'ref.txt'), 'w') as ref_f,\
                open(join(tmp_dir, 'dec.txt'), 'w') as dec_f:
            ref_f.write('\n'.join(map(read_file(ref_dir), refs)) + '\n')
            dec_f.write('\n'.join(map(read_file(dec_dir), decs)) + '\n')

        cmd = 'java -Xmx2G -jar {} {} {} -l en -norm'.format(
            _METEOR_PATH, join(tmp_dir, 'dec.txt'), join(tmp_dir, 'ref.txt'))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output
