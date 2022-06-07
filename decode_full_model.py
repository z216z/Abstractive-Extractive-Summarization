""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import argparse
import json
import os
from os.path import join
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op
from toolz.sandbox import unzip
from cytoolz import identity, concat, curry

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp
from statistics import mean
from data.batcher import tokenize
from metric import compute_rouge_n, compute_rouge_l_summ
from decoding import Abstractor, RLExtractor, DecodeDataset

DATASET_PATH = None
DATA_DIR = None

def compute_rouges(dec_outs, gold):
    rouge_1 = compute_rouge_n(list(concat(dec_outs)),list(concat(gold)), n=1))
    rouge_2 = compute_rouge_n(list(concat(dec_outs)),list(concat(gold)), n=2))
    rouge_L = compute_rouge_l_summ(dec_outs, gold)
    return rouge_1, rouge_2, rouge_L

def decode(model_dir, batch_size, max_len, cuda):
    split = 'test' 
    model_dir = join(DATASET_PATH, model_dir)
    save_path = join(model_dir, 'eval')
    start = time()
    # setup model
    with open(join(model_dir, 'rl', 'meta.json')) as f:
        meta = json.loads(f.read())
    if meta['net_args']['abstractor'] is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        abstractor = identity
    else:
        abstractor = Abstractor(join(model_dir, 'abs'),
                                max_len, cuda)
    extractor = RLExtractor(join(model_dir, 'ext'), cuda=cuda)

    # setup loader
    def coll(data):
        source_lists, target_lists = unzip(data)
        sources = list(filter(bool, map(tokenize(None), source_lists)))
        targets = list(filter(bool, map(tokenize(None), target_lists)))
        return sources, targets
    dataset = DecodeDataset(DATA_DIR, split)

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=coll
    )

    # prepare save paths and logs
    os.makedirs(join(save_path, 'output'))
    dec_log = {}
    dec_log['abstractor'] = meta['net_args']['abstractor']
    dec_log['extractor'] = meta['net_args']['extractor']
    dec_log['rl'] = True
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)

    # Decoding
    i = 0
    rouges = {'rouge_1': [], 'rouge_2': [], 'rouge_L': []}
    with torch.no_grad():
        for article_batch, gold_batch in loader:
            ext_arts = []
            ext_inds = []
            for raw_art_sents in article_batch:
                ext = extractor(raw_art_sents)[:-1]  # exclude EOE
                if not ext:
                    # use top-5 if nothing is extracted
                    # in some rare cases rnn-ext does not extract at all
                    ext = list(range(5))[:len(raw_art_sents)]
                else:
                    ext = [i.item() for i in ext]
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += [raw_art_sents[i] for i in ext]
            
            dec_outs = abstractor(ext_arts)
            for (j, n), gold in zip(ext_inds, gold_batch):
                decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                with open(join(save_path, 'output/{}.txt'.format(i)), 'w') as f:
                    f.write('\n'.join(decoded_sents))
                i += 1
                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(i, n_data, i/n_data*100,timedelta(seconds=int(time()-start))), end='')
                
                for index, sent in enumerate(gold):
                    gold[index] = sent[1:-1]
                doc_rouges = compute_rouges(dec_outs, gold)
                for value, key in zip(doc_rouges, rouges.keys()):
                    rouges[key].append(value)
    
    for key in rouges.keys():
        print(f'Average ROUGE-{key.split('_')[-1]}: {mean(rouges[key])}')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run decoding of the full model (RL)')
    parser.add_argument('--data', type=str, default='FNS2022', choices={'FNS2022', 'CNN'}, help='Select the dataset.')
    parser.add_argument('--language', type=str, default='English', choices={'English', 'Greek', 'Spanish'}, help='Select the language if you use FNS2022.')
    
    parser.add_argument('--model_dir', action='store', default='model', 
                        help='root of the full model')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=2,
                        help='batch size of faster decoding')
    parser.add_argument('--max_dec_word', type=int, action='store', default=30,
                        help='maximun words to be decoded for the abstractor')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    DATASET_PATH = '/content/NLP_Project/Dataset'
    if args.data == 'FNS2022':
        LANGUAGE = args.language
    else:
        LANGUAGE = 'English'
    DATASET_PATH = os.path.join(DATASET_PATH, args.data, LANGUAGE)
    DATA_DIR = os.path.join(DATASET_PATH, 'preprocess', 'labels')
    
    decode(args.model_dir, args.batch,
           args.max_dec_word, args.cuda)
