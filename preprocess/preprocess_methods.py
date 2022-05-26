import nltk
import csv
import re
import os
import json
from nltk.tokenize import word_tokenize

nltk.download('punkt')
start, end = '<SOS>', '<EOS>'
tokens = [start, end]
# Read files for the regex check:
with open('/content/NLP_Project/preprocess/conversions.csv') as csvfile:
    conversions = [r for r in csv.reader(csvfile)]
with open('/content/NLP_Project/preprocess/numerical_related_conversions.csv') as csvfile:
    numerical_related_conversions = [r for r in csv.reader(csvfile)]
    
def save_json(bow, file_path):
    json_file = open(file_path, "w")
    json.dump(bow, json_file)
    json_file.close()

def read_json(file_path):
    json_file = open(file_path, "r")
    payload = json.loads(json_file.read())
    json_file.close()
    return payload    
   
def filter_sentence(sentence, common_bow):
    return [w for w in sentence.split(" ") if w in common_bow.keys() or w in tokens]

def regex_check(sentence)
    sentence = re.sub('[^a-zA-Z0-9$£€]', ' ', sentence)

    for row in conversions:
        abbreviation = row[0]
        extended_version = row[1]

        sentence = re.sub(f'(^{abbreviation}\s)|(\s{abbreviation}\s)|(\s{abbreviation}$)',
                          " " + extended_version + " ", sentence)

    for row in numerical_related_conversions:
        abbreviation = row[0]
        extended_version = row[1]
        if abbreviation == "$":
            abbreviation = "\$"

        sentence = re.sub(f'(^{abbreviation}\s)|(\s{abbreviation}\s)|(\s{abbreviation}$)',
                          " " + extended_version + " ", sentence)

        match_at_the_beginning = re.search(f'(^{row[0]}\d)', sentence)
        if match_at_the_beginning is not None:
            sentence = extended_version + " " + sentence[match_at_the_beginning.end() - 1:]

        match_at_the_end = re.search(f'(\d{row[0]}$)', sentence)
        if match_at_the_end is not None:
            sentence = sentence[:match_at_the_end.start() + 1] + " " + extended_version

        matches_in_the_middle = re.finditer(f'(\d{row[0]}\s)|(\s{row[0]}\d)', sentence)
        next_match_delay = 0
        for m in matches_in_the_middle:
            if m is not None:
                match_start = m.start() + next_match_delay
                match_end = m.end() + next_match_delay
                sentence = sentence[:match_start + 1] + " " + extended_version + " " + sentence[match_end - 1:]
                next_match_delay = len(extended_version) - len(abbreviation)

    sentence = re.sub(f'\s+', " ", sentence)
    sentence = re.sub(f'^\s|\s$', "", sentence)
    return sentence
    
def tokenize_sentence(sentence, common_bow=None):
    sentence = sentence.lower()
    # Use regex to transform abbreviations and acronyms into their corresponding extended form:
    sentence = regex_check(sentence)
    if common_bow is not None:
        return filter_sentence(sentence, common_bow)
    return word_tokenize(sentence)

def filter_corpus(corpus, path_tokenized, common_bow):
    with open(path_tokenized, 'w') as fw:
        with open(corpus) as fr:
            for line in fr.readlines():
                filtered_line = filter_sentence(line.strip(), common_bow)
                if len(filtered_line) > 0:
                    fw.write(' '.join(filtered_line) + '\n')

def tokenizer(path_raw, path_tokenized, common_bow):
    with open(path_tokenized, 'w') as fw:
        with open(path_raw) as fr:
            text = ''
            for line in fr.readlines():
                text += f'{line.strip()} '
            sentences = nltk.sent_tokenize(text)
        for s in sentences:
            tokenized_sentence = tokenize_sentence(s, common_bow)
            if len(tokenized_sentence) > 0:
                tokenized_sentence.insert(0, tokens[0])
                tokenized_sentence.append(tokens[1])
                fw.write(' '.join(tokenized_sentence) + '\n')

def generate_corpus(path_raw, path_tokenized):
    with open(path_tokenized, 'w') as fw:
        for _, folder in enumerate(os.listdir(path_raw)):
            for i, file_name in enumerate(os.listdir(os.path.join(path_raw, folder))):
                with open(os.path.join(path_raw, folder, file_name)) as fr:
                    text = ''
                    for line in fr.readlines():
                        text += f'{line.strip()} '
                    sentences = nltk.sent_tokenize(text)
                for s in sentences:
                    tokenized_sentence = tokenize_sentence(s)
                    if len(tokenized_sentence) > 0:
                        tokenized_sentence.insert(0, tokens[0])
                        tokenized_sentence.append(tokens[1])
                        fw.write(' '.join(tokenized_sentence) + '\n')

def generate_bow(path_corpus, vocab_limit=20000):
    with open(path_corpus) as fr:
        words = []
        for line in fr.readlines():
            words += [w for w in line.strip().split(' ') if w not in tokens]
    '''
    words = nltk.corpus.words(path_corpus)
    words = [w for w in words if w not in tokens]
    '''
    bow = nltk.FreqDist(words)
    common_bow = dict(bow.most_common(vocab_limit))
    return bow, common_bow
