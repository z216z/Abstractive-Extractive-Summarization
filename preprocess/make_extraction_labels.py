"""produce the dataset with (psudo) extraction label"""
import os
import json
import shutil
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from cytoolz import compose
from tqdm import tqdm
import sys
sys.path.insert(0,'..')
from NLP_Project import metric

def _split_words(texts):
    return map(lambda t: t.split(), texts)

def get_extract_label(art_sents, abs_sents):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    scores = []
    indices = list(range(len(art_sents)))
    for abst in abs_sents:
        # for each sentence in the abstract, compute the rouge 
        # with all the sentences in the article:
        rouges = list(map(metric.compute_rouge_l(reference=abst, mode='f'),
                          art_sents))
        # Take the index of the article sentence maximizing the score:
        ext = max(indices, key=lambda i: rouges[i])
        indices.remove(ext)
        extracted.append(ext)
        scores.append(rouges[ext])
        if not indices:
            break
    return extracted, scores

def reduce_article_size(art_sents, abs_sents, top_M):
    """ greedily extract top article sentences based on summary sentences """
    arts_indexes = []
    indices = set(range(len(art_sents)))
    for abst in abs_sents:
        # for each sentence in the abstract, compute the rouge 
        # with all the sentences in the article:
        rouges = list(map(metric.compute_rouge_l(reference=abst, mode='f'),
                          art_sents))
        # Take the index of the article sentence maximizing the score:
        sorted_indices = sorted(indices, reverse=True, key=lambda i: rouges[i])
        top_indices = sorted_indices[:top_M]
        arts_indexes += top_indices
        indices.difference_update(top_indices)
        if not indices:
            break
    return arts_indexes

def label(DATASET_PATH, split, art_max_len=None):
    data = {}
    path_reports = os.path.join(DATASET_PATH, 'preprocess', split, 'annual_reports')
    path_summaries = os.path.join(DATASET_PATH, 'preprocess', split, 'gold_summaries')
    split = 'train' if split == 'training' else 'test'
    path_labels = os.path.join(DATASET_PATH, 'preprocess', 'labels', split)
    if not os.path.exists(path_labels):
        os.makedirs(path_labels)
        
    for file_name in tqdm(os.listdir(path_reports)):
        with open(os.path.join(path_reports, file_name)) as fr:
            whole_article = fr.readlines()
        abs_name = file_name.split('.')[0] + '_1.txt'
        with open(os.path.join(path_summaries, abs_name)) as fr:
            data['abstract'] = fr.readlines()
        tokenize = compose(list, _split_words)
        art_sents = tokenize(whole_article)
        abs_sents = tokenize(data['abstract'])
        """
        if art_max_len is not None and len(whole_article) > art_max_len:
            top_M = int(art_max_len/len(data['abstract']))
            top_sentences_indexes = reduce_article_size(art_sents, abs_sents, top_M)
            data['article'] = [art for i, art in enumerate(whole_article) if i in top_sentences_indexes]
            art_sents = tokenize(data['article'])
        else:
        """
        data['article'] = whole_article
        extracted, scores = get_extract_label(art_sents, abs_sents)
        data['extracted'] = extracted
        data['score'] = scores
        with open(os.path.join(path_labels, '{}.json'.format(file_name.split('.')[0])), 'w') as f:
            json.dump(data, f, indent=4)

def split_data(DATASET_PATH):
    val_labels = os.path.join(DATASET_PATH, 'val')
    if not os.path.exists(val_labels):
        os.makedirs(val_labels)
    file_names = os.listdir(os.path.join(DATASET_PATH, 'train'))
    _, X_val, _, _ = train_test_split(file_names, file_names, test_size=0.2, random_state=42)
    for file_name in X_val:
        shutil.move(os.path.join(DATASET_PATH, 'train', file_name), val_labels)

def analyze_documents(DATASET_PATH, split='training'):
    data = {}
    total_len = 0
    rows_distribution = defaultdict(lambda: 0)
    percentage_distribution = defaultdict(lambda: 0)
    weighted_percentage_distribution = defaultdict(lambda: 0)
    path_reports = os.path.join(DATASET_PATH, 'preprocess', 'distribution', split, 'annual_reports')
    path_summaries = os.path.join(DATASET_PATH, 'preprocess', 'distribution', split, 'gold_summaries')
    path_analysis = os.path.join(DATASET_PATH, 'preprocess', 'distribution', 'analysis')
    if not os.path.exists(path_analysis):
        os.makedirs(path_analysis)
    
    for file_name in tqdm(os.listdir(path_reports)):
        with open(os.path.join(path_reports, file_name)) as fr:
            article = fr.readlines()
        abs_name = file_name.split('.')[0] + '_1.txt'
        with open(os.path.join(path_summaries, abs_name)) as fr:
            abstract = fr.readlines()
        tokenize = compose(list, _split_words)
        art_sents = tokenize(article)
        abs_sents = tokenize(abstract)
        scores = _get_scores(art_sents, abs_sents)
        bucket_scores = _get_bucket_scores(scores)
        len_scores = len(scores)
        total_len += len_scores
        data['score'] = scores
        data['bucket'] = bucket_scores
        data['length'] = len_scores
        for key, value in scores.items():
            rows_distribution[key] += value
        for key, value in bucket_scores.items():
            percentage_distribution[key] += value
            weighted_percentage_distribution[key] += value*len_scores
        with open(os.path.join(path_analysis, '{}.json'.format(file_name.split('.')[0])), 'w') as f:
            json.dump(data, f, indent=4)
    
    for key, value in weighted_percentage_distribution.items():
        weighted_percentage_distribution[key] = value/total_len
    with open(os.path.join(path_analysis, 'rows_distribution.json'), 'w') as f:
            json.dump(rows_distribution, f, indent=4)
    with open(os.path.join(path_analysis, 'percentage_distribution.json'), 'w') as f:
            json.dump(percentage_distribution, f, indent=4)
    with open(os.path.join(path_analysis, 'weighted_percentage_distribution.json'), 'w') as f:
            json.dump(weighted_percentage_distribution, f, indent=4)
            
def _get_scores(art_sents, abs_sents):
    scores = defaultdict(lambda: 0)
    indices = list(range(len(art_sents)))
    for abst in abs_sents:
        rouges = list(map(metric.compute_rouge_l(reference=abst, mode='f'), art_sents))
        for i in indices:
            scores[i] += rouges[i]
    return scores

def _get_bucket_scores(scores):
    bucket_scores = defaultdict(lambda: 0)
    buckets = np.array_split(np.array(scores.keys()), 100)
    for p, bucket in enumerate(buckets, 1):
        for i in bucket:
            bucket_scores[p] += scores[i]
        bucket_scores[p] = bucket_scores[p]/len(bucket)
    return bucket_scores
