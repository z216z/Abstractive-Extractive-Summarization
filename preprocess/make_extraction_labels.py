"""produce the dataset with (psudo) extraction label"""
import os
import json
import shutil
from sklearn.model_selection import train_test_split
from cytoolz import compose
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

def label(DATASET_PATH, split):
    data = {}
    path_reports = os.path.join(DATASET_PATH, 'preprocess', split, 'annual_reports')
    path_summaries = os.path.join(DATASET_PATH, 'preprocess', split, 'gold_summaries')
    split = 'train' if split == 'training' else 'val'
    path_labels = os.path.join(DATASET_PATH, 'preprocess', 'labels', split)
    if not os.path.exists(path_labels):
        os.makedirs(path_labels)
        
    for i, file_name in enumerate(os.listdir(path_reports)):
        with open(os.path.join(path_reports, file_name)) as fr:
            data['article'] = fr.readlines()
        abs_name = file_name.split('.')[0] + '_1.txt'
        with open(os.path.join(path_summaries, abs_name)) as fr:
            data['abstract'] = fr.readlines()
        tokenize = compose(list, _split_words)
        art_sents = tokenize(data['article'])
        abs_sents = tokenize(data['abstract'])
        extracted, scores = get_extract_label(art_sents, abs_sents)
        data['extracted'] = extracted
        data['score'] = scores
        with open(os.path.join(path_labels, '{}.json'.format(file_name.split('.')[0])), 'w') as f:
            json.dump(data, f, indent=4)

""" Method not used: we use the training-validation split of the competition
def split_data(DATASET_PATH):
    path_labels = os.path.join(DATASET_PATH, 'preprocess', 'labels')
    file_names = os.listdir(os.path.join(path_labels, 'all'))
    X_train, X_test, y_train, y_test = train_test_split(file_names, file_names, test_size=0.3, random_state=42)
    for file_name in X_train:
        shutil.copyfile(os.path.join(path_labels, 'all', file_name), os.path.join(path_labels, 'training', file_name))
    for file_name in X_test:
        shutil.copyfile(os.path.join(path_labels, 'all', file_name), os.path.join(path_labels, 'validation', file_name))
"""
