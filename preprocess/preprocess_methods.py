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
    
def _filter_sentence(sentence, common_bow):
    return [w for w in sentence.split(" ") if w in common_bow.keys() or w in tokens]
    
def _tokenize_sentence(sentence, language, common_bow=None):
    sentence = sentence.lower()
    # Use regex to transform abbreviations and acronyms into their corresponding extended form:
    sentence = regex_check(sentence, language, use_abbreviations=True)
    if common_bow is not None:
        return _filter_sentence(sentence, common_bow)
    return word_tokenize(sentence)

def filter_corpus(corpus, path_tokenized, common_bow):
    with open(path_tokenized, 'w') as fw:
        with open(corpus) as fr:
            for line in fr.readlines():
                filtered_line = _filter_sentence(line.strip(), common_bow)
                if len(filtered_line) > 0:
                    fw.write(' '.join(filtered_line) + '\n')

def cut_document(path_raw, max_len, language, distribution=False):
    with open(path_raw, 'r+') as fr:
        text = ''
        for line in fr.readlines():
            text += f'{line.strip()} '
        sentences = nltk.sent_tokenize(text)
        filtered_sentences = []
        if distribution:
            if language=="English":
                if len(sentences) < 500:
                    # take first 30%
                    filtered_sentences = sentences[:len(sentences)/100*30]
                elif len(sentences) >= 500 and len(sentences) < 1000:
                    # take first 27%
                    filtered_sentences = sentences[:len(sentences)/100*27]
                else:
                    # take first 3-14% if it's less than 501 rows, 
                    # otherwise take first 4-11% if it's less than 501 rows, 
                    # otherwise take from 4% until we reach 500 rows
                    temp_sentences = sentences[len(sentences)/100*2:len(sentences)/100*14]
                    if len(temp_sentences) > 500:
                        temp_sentences = sentences[len(sentences)/100*3:len(sentences)/100*11]
                    if len(temp_sentences) > 500:
                        temp_sentences = sentences[len(sentences)/100*3:len(sentences)/100*3+500]
                    filtered_sentences = temp_sentences
            elif language=="Greek":
                if len(sentences) < 500:
                    filtered_sentences += sentences[:len(sentences)/100*4]
                    filtered_sentences += sentences[len(sentences)/100*10:len(sentences)/100*28]
                    filtered_sentences += sentences[len(sentences)/100*73:len(sentences)/100*92]
                    filtered_sentences += sentences[len(sentences)/100*94:]
                elif len(sentences) >= 500 and len(sentences) < 1000:
                    filtered_sentences += sentences[:len(sentences)/100*3]
                    filtered_sentences += sentences[len(sentences)/100*6:len(sentences)/100*7]
                    filtered_sentences += sentences[len(sentences)/100*8:len(sentences)/100*24]
                    filtered_sentences += sentences[len(sentences)/100*27:len(sentences)/100*28]
                    filtered_sentences += sentences[len(sentences)/100*36:len(sentences)/100*37]
                    filtered_sentences += sentences[len(sentences)/100*51:len(sentences)/100*54]
                    filtered_sentences += sentences[len(sentences)/100*63:len(sentences)/100*64]
                    filtered_sentences += sentences[len(sentences)/100*65:len(sentences)/100*66]
                    filtered_sentences += sentences[len(sentences)/100*67:len(sentences)/100*74]
                    filtered_sentences += sentences[len(sentences)/100*78:]
                else:
                    temp_sentences = []
                    # take as sure the first buckets and last buckets, after take the needed sentences to reach 500 sentences
                    start_sentences_important_1 = sentences[:len(sentences)/100*1]
                    start_sentences_important_3 = sentences[len(sentences)/100*9:len(sentences)/100*13]
                    end_sentences = sentences[len(sentences)/100*96:]
                    
                    start_sentences_2 = sentences[len(sentences)/100*1:len(sentences)/100*5]
                    start_sentences_2 += sentences[len(sentences)/100*6:len(sentences)/100*9]
                    temp_sentences += sentences[len(sentences)/100*15:len(sentences)/100*16]
                    temp_sentences += sentences[len(sentences)/100*35:len(sentences)/100*36]
                    temp_sentences += sentences[len(sentences)/100*58:len(sentences)/100*62]
                    temp_sentences += sentences[len(sentences)/100*64:len(sentences)/100*66]
                    temp_sentences += sentences[len(sentences)/100*69:len(sentences)/100*76]
                    temp_sentences += sentences[len(sentences)/100*77:len(sentences)/100*87]
                    temp_sentences += sentences[len(sentences)/100*91:len(sentences)/100*93]
                    temp_sentences += sentences[len(sentences)/100*94:len(sentences)/100*96]
                    
                    mid_sentences = []
                    if 500-len(start_sentences)-len(end_sentences) > 0:
                        mid_sentences = temp_sentences[:500-len(start_sentences)-len(end_sentences)]
                    
                    filtered_sentences = start_sentences_important_1 + start_sentences_2 + start_sentences_important_3 + mid_sentences + end_sentences
            elif language=="Spanish":
                # For Spanish, the distribution says that, for files having more than 500 rows,
                # the most important buckets are the first ones, while the others have all the same importance.
                # In this case we take only the first 500 rows
                if len(sentences) < 500:
                    filtered_sentences += sentences[:len(sentences)/100*31]
                    filtered_sentences += sentences[len(sentences)/100*47:len(sentences)/100*48]
                    filtered_sentences += sentences[len(sentences)/100*59:len(sentences)/100*60]
                else:
                    filtered_sentences = sentences[:500]
        else:
           if len(sentences) > max_len:
              for s in sentences:
                  # delete numbered lists (e.g. '10.', 'A.')
                  if len(s) > 3:
                      filtered_sentences.append(s)
                  if len(filtered_sentences) == max_len:
                      break
        fr.seek(0)
        fr.write('\n'.join(filtered_sentences))
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

def generate_corpus(path_raw, path_tokenized, language, mode='w'):
    with open(path_tokenized, mode=mode) as fw:
        for _, folder in enumerate(os.listdir(path_raw)):
            for i, file_name in enumerate(os.listdir(os.path.join(path_raw, folder))):
                with open(os.path.join(path_raw, folder, file_name)) as fr:
                    text = ''
                    for line in fr.readlines():
                        text += f'{line.strip()} '
                    sentences = nltk.sent_tokenize(text)
                for s in sentences:
                    tokenized_sentence = _tokenize_sentence(s, language)
                    if len(tokenized_sentence) > 3:
                        # if we are using the multilingual model, we don't need eos and sos tokens
                        if mode=='w':
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
