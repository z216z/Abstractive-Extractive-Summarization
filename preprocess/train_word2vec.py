import os
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def train_word2vec(DATASET_PATH, CORPUS_FILTERED_PATH, emb_dim):
    """pretrain a word2vec on the corpus"""
    corpus=[]
    with open(CORPUS_FILTERED_PATH) as fr:
        for line in fr.readlines():
            corpus.append(line.strip().split(" "))
    w2v = Word2Vec(corpus, sg=1, min_count=3, window=2, size=emb_dim, sample=6e-5, alpha=0.05, negative=20, epochs=15)
    w2v.save(os.path.join(DATASET_PATH, 'preprocess', 'word2vec.model'))      

def train_doc2vec(DATASET_PATH, CORPUS_FILTERED_PATH, emb_dim):
    """pretrain a doc2vec on the corpus"""
    corpus=[]
    with open(CORPUS_FILTERED_PATH) as fr:
        for i, line in enumerate(fr.readlines()):
            corpus.append(TaggedDocument(line.strip().split(" "), [str(i)])
    d2v = Doc2Vec(documents=corpus, vector_size=emb_dim, min_count=3, dm=1, window=2, size=emb_dim, sample=6e-5, alpha=0.05, negative=20, epochs=15)
    d2v.train(corpus, total_examples=d2v.corpus_count, epochs=d2v.epochs)
    d2v.save(os.path.join(DATASET_PATH, 'preprocess', 'doc2vec.model'))
