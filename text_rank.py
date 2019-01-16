""" 
TextRank for key word extration and summarization 
code from https://blog.csdn.net/oxuzhenyi/article/details/54981372
"""
import argparse
import os
from os.path import join, exists
import codecs
import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import math
from itertools import product, count
from string import punctuation
from heapq import nlargest

from data_analysis import count_data_by_suffix
from decoding import make_html_safe


"""
传入两个句子
返回这两个句子的相似度
"""


def calculate_similarity(sen1, sen2):
    # 设置counter计数器
    counter = 0.0
    # 长度短的句子不计算
    if len(sen1) <= 3 or len(sen2) <= 3:
        return counter

    for word in sen1:
        if word in sen2:
            counter += 1
    return counter / (math.log(len(sen1)) + math.log(len(sen2)) + 1e-6)


"""
传入句子的列表
返回各个句子之间相似度的图
（邻接矩阵表示）
"""


def create_graph(word_sent):
    num = len(word_sent)
    # 初始化表
    board = [[0.0 for _ in range(num)] for _ in range(num)]

    for i, j in product(range(num), repeat=2):
        if i != j:
            board[i][j] = calculate_similarity(word_sent[i], word_sent[j])
    return board


"""
输入相似度邻接矩阵
返回各个句子的分数
"""


def weighted_pagerank(weight_graph):
    # 把初始的分数值设置为0.5
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]

    # 开始迭代
    while different(scores, old_scores):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]

        for i in range(len(weight_graph)):
            scores[i] = calculate_score(weight_graph, scores, i)
    return scores


"""
判断前后分数有没有变化
这里认为前后差距小于1e-3
分数就趋于稳定
"""


def different(scores, old_scores):
    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= 1e-3:
            flag = True
            break
    return flag


"""
根据公式求出指定句子的分数
"""


def calculate_score(weight_graph, scores, i):
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0

    for j in range(length):
        fraction = 0.0
        denominator = 0.0
        # 先计算分子
        fraction = weight_graph[j][i] * scores[j]
        # 计算分母
        for k in range(length):
            denominator += weight_graph[j][k]
        added_score += fraction / (denominator + 1e-6)
    # 算出最终的分数
    weighted_score = (1 - d) + d * added_score

    return weighted_score


def summarize(origin_sents, stop_words, n=3):
    # 首先分出句子
    # sents = sent_tokenize(text)
    # 然后分出单词
    # word_sent是一个二维的列表
    # word_sent[i]代表的是第i句
    # word_sent[i][j]代表的是
    # 第i句中的第j个单词
    # word_sent = [word_tokenize(s.lower()) for s in sents]

    article_sents = list(map(lambda x: x.split(), origin_sents))
    # 把停用词去除
    for i in range(len(article_sents)):
        for word in article_sents[i]:
            if word in stop_words:
                article_sents[i].remove(word)
    similarity_graph = create_graph(article_sents)
    scores = weighted_pagerank(similarity_graph)
    sent_selected = nlargest(n, zip(scores, count()))
    sent_index = []
    for i in range(n):
        try:
            sent_index.append(sent_selected[i][1])
        except:
            print(len(sent_selected), n)

    sent_res = [origin_sents[i] for i in sent_index]
    return sent_res, sent_index


def main():
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
    stop_words = set(stopwords.words('english') + list(punctuation))

    n_sent = 3
    txt_file = codecs.open(output_path, "w", encoding='utf-8')
    idx_file = codecs.open("%s.idx" % output_path, "w", encoding='utf-8')
    for i in tqdm(range(n_data)):
        # if i > 10:
        #     break
        with open(join(input_path, '{}.{}'.format(i, suffix))) as f:
            data = json.loads(f.read())
            article_sents = list(
                map(lambda x: make_html_safe(x), data['article']))

            summary, idx_sum = summarize(article_sents, stop_words, n=n_sent)
            # import pdb
            # pdb.set_trace()
            summary_tag = " ".join(
                map(lambda x: "<t> %s </t>" % x, summary))
            txt_file.write(summary_tag + "\n")
            idx_file.write(", ".join(map(str, idx_sum)) + "\n")

    txt_file.close()
    idx_file.close()


if __name__ == '__main__':
    main()
