import nltk
import csv
import re
import os
from nltk.tokenize import word_tokenize

nltk.download('punkt')
with open('regex.csv') as csvfile:
    regex = csv.reader(csvfile)

def tokenize_sentence(sentence, common_bow=None):
    sentence = sentence.lower()
    for row in regex: 
        sentence = re.sub(row[0], row[1], sentence)
    if common_bow not None:
        return filter_sentence(sentence, common_bow)
    return word_tokenize.tokenize(sentence)

def filter_sentence(sentence, common_bow):
    return [w for w in sentence if w in common_bow.keys()]

def tokenizer(path_raw, path_tokenized, start='<SOS>', end='<EOS>', common_bow=None, regex=True):
    with open(path_tokenized, 'w') as fw:
        for i, file_name enumerate(os.listdir(path_raw)):
            with open(os.path.join(path_raw, file_name)) as fr:
                text = ''
                for line in fr.readlines():
                    text += f'{line.strip()} '
                sentences = nltk.sent_tokenize(text)
            for s in sentences:
                if regex:
                    tokenized_sentence = tokenize_sentence(s, common_bow)
                elif common_bow not None and not regex:
                    tokenized_sentence = filter_sentence(s, common_bow)
                else:
                    print("Error: Common bow parameter missing and control setting setted to false.")
                if len(tokenized_sentence) > 0:
                    tokenized_sentence.insert(0, start)
                    tokenized_sentence.append(end)
                    fw.write(' '.join(tokenized_sentence) + '\n')

def generate_bow(path_corpus, vocab_limit=20000, start='<SOS>', end='<EOS>'):
    '''
    with open(path_corpus) as fr:
        words = []
        for line in fr.readlines().strip():
            for w in line.split(' '):
                words.append(w)
    '''
    words = nltk.corpus.words(path_corpus)
    words = [w for w in words if w not in [start, end]]
    bow = nltk.FreqDist(words)
    common_bow = dict(bow.most_common(vocab_limit))
    return bow, common_bow
