import os
import sys
import json
import argparse

from make_extraction_labels import label, split_data
from preprocess_methods import generate_corpus, tokenizer, generate_bow, filter_corpus, save_json, read_json
from train_word2vec import train_word2vec

stage_help = 'Select the starting preprocess stage: \n' \
'0 - Generate Corpus \n' \
'1 - Generate BoW \n' \
'2 - Filter Corpus \n' \
'3 - Process Reports and Summaries \n' \
'4 - Train Word2Vec \n' \
'5 - Generate Labels.'

def pipeline(DATASET_PATH, LANGUAGE, STAGE):    
    CORPUS_TOKENIZED_PATH = os.path.join(DATASET_PATH, 'preprocess', 'corpus_tokenized.txt')
    CORPUS_FILTERED_PATH = os.path.join(DATASET_PATH, 'preprocess', 'corpus_filtered.txt')
    
    if STAGE == 0:
        generate_corpus(os.path.join(DATASET_PATH, 'training'), CORPUS_TOKENIZED_PATH, LANGUAGE)
        print('Corpus generated!')
        STAGE = 1
    
    if STAGE == 1 and os.path.exists(CORPUS_TOKENIZED_PATH):
        bow, common_bow = generate_bow(CORPUS_TOKENIZED_PATH)
        save_json(bow, os.path.join(DATASET_PATH, 'preprocess', 'bow.json'))
        save_json(common_bow, os.path.join(DATASET_PATH, 'preprocess', 'common_bow.json'))
        print('BoW generated!')
        STAGE = 2
    
    if STAGE == 2 and os.path.exists(CORPUS_TOKENIZED_PATH):
        try:
            common_bow = read_json(os.path.join(DATASET_PATH, 'preprocess', 'common_bow.json'))
        except ValueError:
            return False
        filter_corpus(CORPUS_TOKENIZED_PATH, CORPUS_FILTERED_PATH, common_bow)
        print('Corpus filtered!')
        STAGE = 3
    
    if STAGE == 3 and os.path.exists(CORPUS_TOKENIZED_PATH) and os.path.exists(CORPUS_FILTERED_PATH):
        try:
            common_bow = read_json(os.path.join(DATASET_PATH, 'preprocess', 'common_bow.json'))
        except ValueError:
            return False
        for _, folder in enumerate(os.listdir(os.path.join(DATASET_PATH, 'training'))): # folders = [annual_reports, golden_summaries]
            for i, file_name in enumerate(os.listdir(os.path.join(DATASET_PATH, 'training', folder))):
                tokenizer(os.path.join(DATASET_PATH, 'training', folder, file_name), os.path.join(DATASET_PATH, 'preprocess', folder, file_name), LANGUAGE, common_bow)
            print(f'{folder} processed!')
        STAGE = 4
    
    if STAGE == 4 and os.path.exists(CORPUS_FILTERED_PATH):
        train_word2vec(DATASET_PATH, CORPUS_FILTERED_PATH)
        print('Word2Vec model saved!')
        STAGE = 5
    
    if STAGE == 5 and len(os.listdir(os.path.join(DATASET_PATH, 'preprocess', 'annual_reports'))) > 0 and len(os.listdir(os.path.join(DATASET_PATH, 'preprocess', 'gold_summaries'))) > 0:
        label(DATASET_PATH)
        print('Labels generated!')
        split_data(DATASET_PATH)
        print('Labels partitioned!')
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Run the preprocess pipeline.'
    )
    parser.add_argument('--data', type=str, default='FNS2022', choices={'FNS2022', 'CNN'}, help='Select the dataset.')
    parser.add_argument('--language', type=str, default='English', choices={'English', 'Greek', 'Spanish'}, help='Select the language if you use FNS2022.')
    parser.add_argument('--stage', type=int, default=0, choices={0, 1, 2, 3, 4, 5}, help=stage_help)
    args = parser.parse_args()
    
    DATASET_PATH = '/content/NLP_Project/Dataset'
    if args.data == 'FNS2022':
        DATASET_PATH = os.path.join(DATASET_PATH, args.data, args.language)
        LANGUAGE = args.language
    else:
        DATASET_PATH = os.path.join(DATASET_PATH, args.data)
        LANGUAGE = 'English'
    
    if not os.path.exists(os.path.join(DATASET_PATH, 'preprocess')):
        os.makedirs(os.path.join(DATASET_PATH, 'preprocess', 'annual_reports'))
        os.makedirs(os.path.join(DATASET_PATH, 'preprocess', 'gold_summaries'))
        
    pipeline(DATASET_PATH, LANGUAGE, args.stage)
