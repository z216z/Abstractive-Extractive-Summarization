import nltk
import csv
import re
import os
import json
from nltk.tokenize import word_tokenize
from regexp.utils import regex_check

nltk.download('punkt')
start, end = '<SOS>', '<EOS>'
tokens = [start, end]
    
def save_json(bow, file_path):
    json_file = open(file_path, "w")
    json.dump(bow, json_file)
    json_file.close()

def read_json(file_path):
    json_file = open(file_path, "r")
    payload = json.loads(json_file.read())
    json_file.close()
    return payload    
    
def _tokenize_sentence(sentence, language, common_bow=None):
    def filter_sentence(sentence, common_bow):
        return [w for w in sentence.split(" ") if w in common_bow.keys() or w in tokens]
    sentence = sentence.lower()
    # Use regex to transform abbreviations and acronyms into their corresponding extended form:
    sentence = regex_check(sentence, language, use_abbreviations=True)
    if common_bow is not None:
        return filter_sentence(sentence, common_bow)
    return word_tokenize(sentence)

def filter_corpus(corpus, path_tokenized, common_bow):
    with open(path_tokenized, 'w') as fw:
        with open(corpus) as fr:
            for line in fr.readlines():
                filtered_line = _filter_sentence(line.strip(), common_bow)
                if len(filtered_line) > 0:
                    fw.write(' '.join(filtered_line) + '\n')

def cut_document(path_raw, max_len):
    with open(path_raw, 'r+') as fr:
        text = ''
        for line in fr.readlines():
            text += f'{line.strip()} '
        sentences = nltk.sent_tokenize(text)
        if len(sentences) > max_len:
            sentences = sentences[:max_len]
            fr.seek(0)
            fr.write('\n'.join(sentences))
            fr.truncate()
                  
def tokenizer(path_raw, path_tokenized, language, common_bow):
    with open(path_tokenized, 'w') as fw:
        with open(path_raw) as fr:
            text = ''
            for line in fr.readlines():
                text += f'{line.strip()} '
            sentences = nltk.sent_tokenize(text)
        for s in sentences:
            tokenized_sentence = _tokenize_sentence(s, language, common_bow)
            if len(tokenized_sentence) > 0:
                tokenized_sentence.insert(0, tokens[0])
                tokenized_sentence.append(tokens[1])
                fw.write(' '.join(tokenized_sentence) + '\n')

def generate_corpus(path_raw, path_tokenized, language):
    with open(path_tokenized, 'w') as fw:
        for _, folder in enumerate(os.listdir(path_raw)):
            for i, file_name in enumerate(os.listdir(os.path.join(path_raw, folder))):
                with open(os.path.join(path_raw, folder, file_name)) as fr:
                    text = ''
                    for line in fr.readlines():
                        text += f'{line.strip()} '
                    sentences = nltk.sent_tokenize(text)
                for s in sentences:
                    tokenized_sentence = _tokenize_sentence(s, language)
                    if len(tokenized_sentence) > 0:
                        tokenized_sentence.insert(0, tokens[0])
                        tokenized_sentence.append(tokens[1])
                        fw.write(' '.join(tokenized_sentence) + '\n')

def generate_bow(path_corpus, vocab_limit=20000):
    with open(path_corpus) as fr:
        words = []
        for line in fr.readlines():
            words += [w for w in line.strip().split(' ') if w not in tokens]
    bow = nltk.FreqDist(words)
    common_bow = dict(bow.most_common(vocab_limit))
    return bow, common_bow
