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
from numba import jit

def _split_words(texts):
    return map(lambda t: t.split(), texts)

@jit
def get_extract_label_jit(art_sents, abs_sents):
    """ greedily match summary sentences to article sentences"""
    extracted = np.empty(0)
    scores = np.empty(0)
    indices = np.array(list(range(len(art_sents))))
    for abst in abs_sents:
        # for each sentence in the abstract, compute the rouge 
        # with all the sentences in the article:
        rouges = np.array(list(map(metric.compute_rouge_l_jit(reference=abst, mode='f'),
                          art_sents)))
        # Take the index of the article sentence maximizing the score:
        temp = np.zeros(rouges.size) - 1
        for id in indices:
            temp[id] = rouges[id]
        ext = np.argmax(temp)
        indices = indices[indices != ext]
        extracted = np.append(extracted, ext)
        scores = np.append(scores, np.take(rouges, ext))
        if indices.size == 0:
            break
    extracted = extracted.astype(int)
    return extracted.tolist(), scores.tolist()

def get_extract_label_original(art_sents, abs_sents):
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

def get_extract_label(art_sents, abs_sents, jit=True):
    if jit:
        return get_extract_label_jit(art_sents, abs_sents)
    else:
        return get_extract_label_original(art_sents, abs_sents)

def label(DATASET_PATH, split, jit=True):
    data = {}
    path_reports = os.path.join(DATASET_PATH, 'preprocess', split, 'annual_reports')
    path_summaries = os.path.join(DATASET_PATH, 'preprocess', split, 'gold_summaries')
    split = 'train' if split == 'training' else 'test'
    path_labels = os.path.join(DATASET_PATH, 'preprocess', 'labels', split)
    if not os.path.exists(path_labels):
        os.makedirs(path_labels)
        
    for file_name in tqdm(os.listdir(path_reports)):
        with open(os.path.join(path_reports, file_name)) as fr:
            article = fr.readlines()
        if len(article) > 0:
            abstract = _get_abstract(path_summaries, file_name.split('.')[0], len(article))
            if abstract is not None:
                data['abstract'] = abstract
                tokenize = compose(list, _split_words)
                art_sents = tokenize(article)
                abs_sents = tokenize(data['abstract'])
                data['article'] = article
                extracted, scores = get_extract_label(art_sents, abs_sents, jit)
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

def _get_abstract(path_summaries, file_name, article_len):
    abs_names = [s for s in os.listdir(path_summaries) if s.split('_')[0] == file_name]
    abs_names.sort()
    for abs_name in abs_names:
        with open(os.path.join(path_summaries, abs_name)) as fr:
            abstract = fr.readlines()
        if len(abstract) < article_len and len(abstract) > 0:
            return abstract
    return None
    
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
        if len(article) > 0:
            abstract = _get_abstract(path_summaries, file_name.split('.')[0], len(article))
            if abstract is not None:
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
                for key, value in zip(list(range(len(scores))), scores):
                    rows_distribution[key] += value
                for key, value in zip(list(range(1, 101)), bucket_scores):
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
            
@jit
def _get_scores(art_sents, abs_sents):
    indices = np.array(list(range(len(art_sents))))
    scores = np.zeros(indices.size)
    for abst in abs_sents:
        rouges = np.array(list(map(metric.compute_rouge_l(reference=abst, mode='f'), art_sents)))
        for i in indices:
            scores[i] += rouges[i]
    return scores.tolist()

@jit
def _get_bucket_scores(scores):
    scores = np.array(scores)
    indices = np.array(list(range(len(scores))))
    bucket_scores = np.zeros(100)
    buckets = np.array_split(indices, 100)
    for p, bucket in enumerate(buckets):
        if len(bucket) > 0:
            for i in bucket:
                bucket_scores[p] += scores[i]
            bucket_scores[p] = bucket_scores[p]/len(bucket)
        else:
            bucket_scores[p] = 0
    return bucket_scores.tolist()
