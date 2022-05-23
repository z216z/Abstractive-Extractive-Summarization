import nltk
import csv
import re
import os
from nltk.tokenize import word_tokenize

nltk.download('punkt')
with open('regex.csv') as csvfile:
    regex = csv.reader(csvfile)

def tokenize_sentence(sentence):
    sentence = sentence.lower()
    for row in regex: 
        sentence = re.sub(row[0], row[1], sentence)
    
    return word_tokenize.tokenize(sentence)

def tokenize_corpus(path_reports, path_corpus, start='<SOS>', end='<EOS>'):
	with open(path_corpus, 'w') as fw:
		for i, file_name enumerate(os.listdir(path_reports)):
			with open(os.path.join(path_reports, file_name)) as fr:
                text = ''
                for line in fr.readlines():
                    text += f'{line.strip()} '
                sentences = nltk.sent_tokenize(text)
            for s in sentences:
                tokenized_sentence = tokenize_sentence(s)
                if len(tokenized_sentence) > 0:
                    tokenized_sentence.insert(0, start)
                    tokenized_sentence.append(end)
                    fw.write(' '.join(tokenized_sentence)+'\n')

def generate_bow(path_corpus, vocab_limit=20000):
    '''
    with open(path_corpus) as fr:
        words = []
        for line in fr.readlines().strip():
            for w in line.split(' '):
                words.append(w)
    '''
    words = nltk.corpus.words(path_corpus)
    bow = nltk.FreqDist(words)
    common_bow = dict(bow.most_common(vocab_limit))
    return bow, common_bow

def doc_filter(a):
    return a
def doc_preprocessing(a):
    return a
