"""produce the dataset with (psudo) extraction label"""
import os
import json
from cytoolz import compose
from NLP_Project import metric
from metric import compute_rouge_l

def _split_words(texts):
    return map(lambda t: t.split(), texts)

def get_extract_label(art_sents, abs_sents):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    scores = []
    indices = list(range(len(art_sents)))
    for abst in abs_sents:
        rouges = list(map(compute_rouge_l(reference=abst, mode='r'),
                          art_sents))
        ext = max(indices, key=lambda i: rouges[i])
        indices.remove(ext)
        extracted.append(ext)
        scores.append(rouges[ext])
        if not indices:
            break
    return extracted, scores

def label(DATASET_PATH):
    data = {}
    path_reports = os.path.join(DATASET_PATH, 'preprocess', 'annual_reports')
    path_summaries = os.makedirs(os.path.join(DATASET_PATH, 'preprocess', 'gold_summaries'))
    path_labels = os.makedirs(os.path.join(DATASET_PATH, 'preprocess', 'labels'))
    if not os.path.exists(path_labels):
        os.makedirs(path_labels)
        
    for i, file_name in enumerate(os.listdir(path_reports)):
        with open(os.path.join(path_reports, file_name)) as fr:
            data['article'] = fr.readlines()
        with open(os.path.join(path_summaries, file_name)) as fr:
            data['abstract'] = fr.readlines()
        tokenize = compose(list, _split_words)
        art_sents = tokenize(data['article'])
        abs_sents = tokenize(data['abstract'])
        extracted, scores = get_extract_label(art_sents, abs_sents)
        data['extracted'] = extracted
        data['score'] = scores
        with open(os.path.join(path_labels, '{}.json'.format(file_name)), 'w') as f:
            json.dump(data, f, indent=4)
