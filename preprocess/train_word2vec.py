""" pretrain a word2vec on the corpus"""
import os
import gensim
from gensim.models import Word2Vec

def train_word2vec(DATASET_PATH, CORPUS_FILTERED_PATH, emb_dim):
    corpus=[]
    with open(CORPUS_FILTERED_PATH) as fr:
        for line in fr.readlines():
            corpus.append(line.strip().split(" "))
    w2v = Word2Vec(corpus, sg=1, min_count=3, window=2, size=emb_dim, sample=6e-5, alpha=0.05, negative=20, iter=15)
    w2v.save(os.path.join(DATASET_PATH, 'preprocess', 'word2vec.model'))      
