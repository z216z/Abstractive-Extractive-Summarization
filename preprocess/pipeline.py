import os
import sys
import json
import argparse
from tqdm import tqdm
from make_extraction_labels import label, split_data
from preprocess_methods import *
from train_word2vec import train_word2vec, train_doc2vec

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
        if args.max_len is not None:
            for _, split in enumerate(['training', 'validation']):
                for i, file_name in enumerate(os.listdir(os.path.join(DATASET_PATH, split, 'annual_reports'))):
                    cut_document(os.path.join(DATASET_PATH, split, 'annual_reports', file_name), args.max_len)
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
        for split in ['training', 'validation']:
            for folder in ['annual_reports', 'gold_summaries']:
                for file_name in tqdm(os.listdir(os.path.join(DATASET_PATH, split, folder))):
                    tokenizer(os.path.join(DATASET_PATH, split, folder, file_name), os.path.join(DATASET_PATH, 'preprocess', split, folder, file_name), LANGUAGE, common_bow)
                print(f'{split} {folder} processed!')
        STAGE = 4
    
    if STAGE == 4 and os.path.exists(CORPUS_FILTERED_PATH):
        if LANGUAGE == 'Multi':
            train_doc2vec(DATASET_PATH, CORPUS_FILTERED_PATH, args.emb_dim)
            print('Doc2Vec model saved!')
        else:
            train_word2vec(DATASET_PATH, CORPUS_FILTERED_PATH, args.emb_dim)
            print('Word2Vec model saved!')
        STAGE = 5
    
    if STAGE == 5 and \
        len(os.listdir(os.path.join(DATASET_PATH, 'preprocess', 'training', 'annual_reports'))) > 0 and \
        len(os.listdir(os.path.join(DATASET_PATH, 'preprocess', 'training', 'gold_summaries'))) > 0 and \
        len(os.listdir(os.path.join(DATASET_PATH, 'preprocess', 'validation', 'annual_reports'))) > 0 and \
        len(os.listdir(os.path.join(DATASET_PATH, 'preprocess', 'validation', 'gold_summaries'))) > 0:
        for _, split in enumerate(['training', 'validation']):
            label(DATASET_PATH, split)
            split = 'train' if split == 'training' else 'test'
            print(f'Labels generated for the {split} set!')
        split_data(os.path.join(DATASET_PATH, 'preprocess', 'labels'))
        print(f'Labels generated for the validation set!')
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Run the preprocess pipeline.'
    )
    parser.add_argument('--data', type=str, default='FNS2022', choices={'FNS2022', 'CNN'}, help='Select the dataset.')
    parser.add_argument('--language', type=str, default='English', choices={'English', 'Greek', 'Spanish', 'Multi'}, help='Select the language if you use FNS2022.')
    parser.add_argument('--stage', type=int, default=0, choices={0, 1, 2, 3, 4, 5}, help=stage_help)
    parser.add_argument('--emb_dim', type=int, default=300, action='store', help='The dimension of word embedding.')
    parser.add_argument('--max_len', type=int, default=1000, action='store', help='Limit the number of sentences in the articles for training purposes.')
    args = parser.parse_args()
    
    DATASET_PATH = '/content/NLP_Project/Dataset'
    if args.data == 'FNS2022':
        LANGUAGE = args.language
    else:
        LANGUAGE = 'English'
    
    if LANGUAGE != 'Multi':
        DATASET_PATH = os.path.join(DATASET_PATH, args.data, LANGUAGE)
    else:
        DATASET_PATH = os.path.join(DATASET_PATH, args.data)
    
    if not os.path.exists(os.path.join(DATASET_PATH, 'preprocess')):
        for _, split in enumerate(['training', 'validation']):
            os.makedirs(os.path.join(DATASET_PATH, 'preprocess', split, 'annual_reports'))
            os.makedirs(os.path.join(DATASET_PATH, 'preprocess', split, 'gold_summaries'))
        
    pipeline(DATASET_PATH, LANGUAGE, args.stage)
