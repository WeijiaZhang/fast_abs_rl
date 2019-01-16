"""
TFIDF for key word extration and summarization
origin code from https://blog.csdn.net/oxuzhenyi/article/details/54981326
"""
import argparse
import os
from os.path import join, exists
import codecs
import json
from tqdm import tqdm
import math

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
import multiprocessing as mp

from data_analysis import count_data_by_suffix
from decoding import make_html_safe


############################ TFIDF START ############################
"""
计算出每个词出现的频率
article_sents 是一个已经分好词的列表
返回一个词典freq[],
freq[w]代表了w出现的频率
max_cut 变量限制了在文本中出现重要性过高的词。就像在跳水比赛中会去掉最高分和最低分一样。
我们也需要去掉那些重要性过高和过低的词来提升算法的效果。
同理，min_cut 限制了出现频率过低的词。
"""


def cal_tf(article, stop_words, max_cut=0.9, min_cut=0.1):
    freq = defaultdict(int)
    num_words = 0
    for sent in article:
        for word in sent:
            # 注意stopwords
            if word not in stop_words:
                freq[word] += 1
                num_words += 1

    # 计算TF
    # 得出最高出现频次m
    # import pdb
    # pdb.set_trace()
    # m = float(max(freq.values())) + 1e-3
    m = float(num_words) + 1e-3
    # 所有单词的频次统除m
    for w in list(freq.keys()):
        freq[w] = freq[w] / m
        # 感觉cut有点不是很合理？
        # if freq[w] >= max_cut or freq[w] <= min_cut:
        #     del freq[w]
    return freq


def cal_idf(article_all, stop_words):
    """
    defaultdict和普通的dict
    的区别是它可以设置default值
    参数是int默认值是0
    """
    num_article = len(article_all)
    inverse_freq = defaultdict(int)
    # print('Calculating word idf...')
    # 统计每个词出现的频率
    for article in tqdm(article_all):
        # 辅助IDF计算：确保同一文章中的同一单词只被统计一次
        word_visited = defaultdict(bool)
        for sent in article:
            for word in sent:
                # 注意stopwords
                if (word not in stop_words) and (not word_visited[word]):
                    word_visited[word] = True
                    inverse_freq[word] += 1
        # import pdb
        # pdb.set_trace()
    # 计算IDF
    for word in list(inverse_freq.keys()):
        inverse_freq[word] = math.log(
            float(num_article) / (inverse_freq[word] + 1.0))

    return inverse_freq


def save_idf(word_idf, data_path):
    print('Saving idf to file ...')
    word_idf_sorted = sorted(
        word_idf.items(), key=lambda x: x[1], reverse=True)
    data_file = codecs.open(data_path, 'w', encoding='utf-8')
    for word, idf in word_idf_sorted:
        data = {'word': word, 'idf': idf}
        r = json.dumps(data)
        data_file.write(r + "\n")

    data_file.close()


def load_idf(data_path):
    word_idf = defaultdict(int)
    with codecs.open(data_path, encoding='utf-8') as f:
        for line in f.readlines():
            r = json.loads(line.strip())
            word_idf[r['word']] = r['idf']
    return word_idf


def cal_tfidf(article, word_idf, stop_words):
     # cal tf
    word_tf = cal_tf(article, stop_words)
    # cal tfidf
    word_tfidf = defaultdict(int)
    for word in list(word_tf.keys()):
        word_tfidf[word] = word_tf[word] * word_idf[word]
    return word_tfidf

############################ TFIDF END ############################


############################ LOAD DATA START ############################

def load_one_corpus(data_path, suffix, text_type='article'):
    n_data = count_data_by_suffix(data_path, suffix=suffix)
    articles = []
    text_type_all = ['article', 'abstract']
    for i in tqdm(range(n_data)):
        # if i > 100:
        #     break
        with open(join(data_path, '{}.{}'.format(i, suffix))) as f:
            data = json.loads(f.read())
            if text_type == 'all':
                article_sents = list(
                    map(lambda x: make_html_safe(x).split(), data[text_type_all[0]]))
                abstract_sents = " ".join(
                    map(lambda x: "<t> %s </t>" % make_html_safe(x), data[text_type_all[1]]))
                articles.append((article_sents, abstract_sents))
            else:
                article_sents = list(
                    map(lambda x: make_html_safe(x).split(), data[text_type]))
                articles.append(article_sents)

    return articles


def load_corpus(root_path, splits, suffix, text_type='article'):
    article_all = []
    for split in splits:
        data_path = join(root_path, split)
        articles = load_one_corpus(data_path, suffix, text_type=text_type)
        article_all.extend(articles)
    return article_all

############################ LOAD DATA END ############################


############################ SUMMARIZATION START ############################

def search_sent(article, word, sents_idx):
    search_idx = -1
    for i, sent in enumerate(article):
        if i in sents_idx:
            continue
        if word in sent:
            search_idx = i
            break
    return search_idx


def rank_sent_by_max_key(article, word_tfidf, n_sents=3):
    word_sorted = nlargest(n_sents + 100, word_tfidf, key=word_tfidf.get)
    sents_idx = []
    sent_selected = []

    n_iters = min(n_sents, len(article))
    i = 0
    while len(sents_idx) < n_iters:
        word = word_sorted[i]
        search_idx = search_sent(article, word, sents_idx)
        # in some case, some key words are in the same sentences
        if search_idx != -1:
            sents_idx.append(search_idx)
        i += 1

    sents_idx = sorted(sents_idx)
    for i in sents_idx:
        sent_str = ' '.join(article[i])
        sent_selected.append(sent_str)
    return sent_selected, sents_idx, word_sorted[:20]


def rank_sent_by_avg_key(article, word_tfidf, n_sents=3):
    ranking_idx = defaultdict(int)
    for i, sent in enumerate(article):
        num_words = 0
        for word in sent:
            if word in word_tfidf:
                ranking_idx[i] += word_tfidf[word]
                num_words += 1
        # considering length of sentence
        ranking_idx[i] /= (float(num_words) + 1e-3)

    sents_idx = nlargest(n_sents, ranking_idx, key=ranking_idx.get)
    sents_idx = sorted(sents_idx)
    sent_selected = []
    for i in sents_idx:
        sent_str = ' '.join(article[i])
        sent_selected.append(sent_str)
    return sent_selected, sents_idx


def summarize(args, word_idf, stop_words):
    input_path = args.in_path
    output_path = args.out_path
    suffix = args.suffix
    text_type = args.text_type

    n_sents = args.n_sent_sum
    method = args.method_sum

    article_in = load_one_corpus(input_path, suffix, text_type=text_type)
    txt_file = codecs.open(output_path, "w", encoding='utf-8')
    idx_file = codecs.open("%s.idx" % output_path, "w", encoding='utf-8')
    # word_tfidf_all = defaultdict(int)
    for article in tqdm(article_in):
        word_tfidf = cal_tfidf(article, stop_words, word_idf)
        if method == 'max':
            sent_selected, sents_idx, word_sorted = rank_sent_by_max_key(
                article, word_tfidf, n_sents=n_sents)
        else:
            sent_selected, sents_idx = rank_sent_by_avg_key(
                article, word_tfidf, n_sents=n_sents)
        sents_tag = " ".join(
            map(lambda x: "<t> %s </t>" % x, sent_selected))
        txt_file.write(sents_tag + "\n")
        if method == 'max':
            out_str = ", ".join(map(str, sents_idx)) + ' || nlargest weight word: ' + \
                ', '.join(word_sorted)
            idx_file.write(out_str + "\n")
        else:
            idx_file.write(", ".join(map(str, sents_idx)) + "\n")
    txt_file.close()
    idx_file.close()

############################ SUMMARIZATION END ############################


############################ SIMILARITY START ############################

def cal_sent_simi(sent1, sent2, method='cos'):
    sent1_norm = 0.0
    for w in sent1.values():
        sent1_norm += (w ** 2)
    sent1_norm = math.sqrt(sent1_norm)

    sent2_norm = 0.0
    for w in sent2.values():
        sent2_norm += (w ** 2)
    sent2_norm = math.sqrt(sent2_norm)

    sents_dot = 0.0
    for word, w in sent1.items():
        if word in sent2:
            sents_dot += (w * sent2[word])

    if sents_dot > 1e-6:
        return sents_dot / (sent1_norm + sent2_norm + 1e-6)
    else:
        return 0.0


def get_sent_dict(article, word_tfidf, n_sents=3):
    sent_dict = defaultdict(int)
    for sent in article[:n_sents]:
        for word in sent:
            if word in word_tfidf:
                sent_dict[word] = word_tfidf[word]

    return sent_dict


def get_keyword_dict(word_tfidf, n_keys=50):
    keyword_sorted = nlargest(n_keys, word_tfidf, key=word_tfidf.get)
    keyword_dict = defaultdict(int)
    for word in keyword_sorted:
        keyword_dict[word] = word_tfidf[word]
    return keyword_dict


def cal_article_simi(article1, article2, word_idf, stop_words,
                     method='cos', n_sents=3, n_keys=50):
    word_tfidf_a1 = cal_tfidf(article1, word_idf, stop_words)
    word_tfidf_a2 = cal_tfidf(article2, word_idf, stop_words)

    # leading sentence similarity
    sent_dict_a1 = get_sent_dict(article1, word_tfidf_a1, n_sents=n_sents)
    sent_dict_a2 = get_sent_dict(article2, word_tfidf_a2, n_sents=n_sents)
    sent_simi = cal_sent_simi(sent_dict_a1, sent_dict_a2)

    # keyword similarity
    keyword_dict_a1 = get_keyword_dict(word_tfidf_a1, n_keys=n_keys)
    keyword_dict_a2 = get_keyword_dict(word_tfidf_a2, n_keys=n_keys)
    keyword_simi = cal_sent_simi(keyword_dict_a1, keyword_dict_a2)

    return sent_simi, keyword_simi


def generate_template(args, word_idf, stop_words, is_mp=False, pool_size=8):
    root_path = args.root_path
    input_path = args.in_path
    suffix = args.suffix
    text_type = args.text_type

    train_path = join(root_path, 'train')
    text_type_train = 'all'
    article_train = load_one_corpus(
        train_path, suffix, text_type=text_type_train)

    article_in = load_one_corpus(input_path, suffix, text_type=text_type)

    def generate_template_one(art_idx):
        output_path = args.out_path
        n_templates = args.n_templates
        method = args.method_simi
        n_sents = args.n_sent_simi
        n_keys = args.n_keys_simi
        key_weight = args.key_weight_simi

        txt_path = join(output_path, '%s.json' % art_idx)
        txt_file = codecs.open(txt_path, "w", encoding='utf-8')
        article, abstract = article_in[art_idx]
        simi_scores = []
        for j, candidate in enumerate(article_train):
            candidate_art, candidate_abs = candidate
            sent_simi, keyword_simi = cal_article_simi(article, candidate_art, word_idf, stop_words,
                                                       method=method, n_sents=n_sents, n_keys=n_keys)
            article_simi = (1 - key_weight) * sent_simi + \
                key_weight * keyword_simi
            simi_scores.append((j, sent_simi, keyword_simi, article_simi))

        simi_scores_largest = nlargest(
            n_templates, simi_scores, key=lambda x: x[-1])

        temp_dict = dict()
        temp_dict['summary'] = abstract
        for k, ele in enumerate(simi_scores_largest):
            idx = ele[0]
            temp_dict['template_%s' % k] = article_train[idx][1]
            temp_dict['score_%s' %
                      k] = 'idx: %d, sent: %.4f, key: %.4f, article: %.4f' % ele

        txt_file.write(json.dumps(temp_dict, indent=4) + '\n')
        txt_file.close()

    if is_mp:
        with mp.Pool(pool_size) as pool:
            res = pool.map(generate_template_one,
                           list(range(len(article_in))))
    else:
        for idx in tqdm(range(len(article_in))):
            # if idx > 10:
            #     break
            generate_template_one(idx)

############################ SIMILARITY END ############################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--in_path', required=True,
                        help='Path to input path')
    parser.add_argument('-out', '--out_path', required=True,
                        help='Path to output file')
    parser.add_argument('-suf', '--suffix', required=True,
                        help='Suffix of file')
    parser.add_argument('-tt', '--text_type', required=False, type=str,
                        default='article', choices=['article', 'abstract', 'all'],
                        help='Text type')
    parser.add_argument('-root', '--root_path', required=False, type=str,
                        default='../dataset/raw/CNN_Daily/fast_abs_rl/finished_files',
                        help='Path to root path')
    parser.add_argument('-idf', '--idf_path', required=False, type=str,
                        default='./output/data_analysis/cnndm_article_idf.txt',
                        help='Path to word idf file')
    parser.add_argument('-n_sum', '--n_sent_sum', required=False, type=int,
                        default=3, help='Number of extracted sentences for summarization')
    parser.add_argument('-mtd_sum', '--method_sum', required=False, type=str,
                        default='max', help='Method to select extracted sentences for summarization')

    parser.add_argument('-n_t', '--n_templates', required=False, type=int,
                        default=20, help='Number of templates sorted by article similarity')
    parser.add_argument('-n_s', '--n_sent_simi', required=False, type=int,
                        default=3, help='Number of top sentences for calculating article similarity')
    parser.add_argument('-n_k', '--n_keys_simi', required=False, type=int,
                        default=50, help='Number of max keywords for calculating article similarity')
    parser.add_argument('-k_w', '--key_weight_simi', required=False, type=float,
                        default=0.5, help='weights of keyword similarity')
    parser.add_argument('-mtd_simi', '--method_simi', required=False, type=str,
                        default='cos', help='Method to calculate article similarity')

    args = parser.parse_args()
    stop_words = set(stopwords.words('english') + list(punctuation))

    # cal idf
    idf_path = args.idf_path
    suffix = args.suffix
    if not exists(idf_path):
        root_path = args.root_path
        splits = ['train', 'val', 'test']
        article_all = load_corpus(
            root_path, splits, suffix, text_type=text_type)
        word_idf = cal_idf(article_all, stop_words)
        save_idf(word_idf, idf_path)
        del article_all
    else:
        word_idf = load_idf(idf_path)

    # extract sentences by key words for summarazation
    # summarize(args, word_idf, stop_words)

    # generating templates
    is_mp = False
    pool_size = 8
    generate_template(args, word_idf, stop_words,
                      is_mp=is_mp, pool_size=pool_size)


if __name__ == '__main__':
    main()
