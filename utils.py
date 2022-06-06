""" utility functions"""
import re
import os
from os.path import basename

import gensim
import torch
from torch import nn


def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


PAD = 0
UNK = 1
START = 2
END = 3
def make_vocab(wc):
    word2id, id2word = {}, {}
    word2id['<pad>'] = PAD
    word2id['<unk>'] = UNK
    word2id['<SOS>'] = START
    word2id['<EOS>'] = END
    id2word[PAD] = '<pad>'
    id2word[UNK] = '<unk>'
    id2word[START] = '<SOS>'
    id2word[END] = '<EOS>'
    i = 4
    for w in wc:
        if w not in ['<SOS>', '<EOS>']:
            word2id[w] = i
            id2word[i] = w
            i += 1
    return word2id, id2word


def make_embedding(id2word, w2v, emb_dim, initializer=None):
    # attrs = basename(w2v_file).split('.')  #word2vec.{dim}d.{vsize}k.bin
    # w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(id2word)
    embedding = nn.Embedding(vocab_size, emb_dim).weight
    if initializer is not None:
        initializer(embedding)

    oovs = []
    with torch.no_grad():
        for i in range(len(id2word)):
            # NOTE: id2word can be list or dict
            if i == START:
                embedding[i, :] = torch.Tensor(w2v['<SOS>'])
            elif i == END:
                embedding[i, :] = torch.Tensor(w2v[r'<EOS>'])
            elif id2word[i] in w2v:
                embedding[i, :] = torch.Tensor(w2v[id2word[i]])
            else:
                oovs.append(i)
    return embedding, oovs
