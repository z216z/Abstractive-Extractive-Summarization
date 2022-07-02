import os
import sys
import json
import argparse
from tqdm import tqdm
from make_extraction_labels import label, split_data
from preprocess_methods import *
from train_word2vec import train_word2vec
import time
from datetime import timedelta

stage_help = 'Select the starting preprocess stage: \n' \
'0 - Generate Corpus \n' \
'1 - Generate BoW \n' \
'2 - Filter Corpus \n' \
'3 - Process Reports and Summaries \n' \
'4 - Train Word2Vec \n' \
'5 - Generate Labels.'

TASK = None

def pipeline(DATASET_PATH, LANGUAGE, STAGE, splits):    
    CORPUS_TOKENIZED_PATH = os.path.join(DATASET_PATH, 'preprocess', 'corpus_tokenized.txt')
    CORPUS_FILTERED_PATH = os.path.join(DATASET_PATH, 'preprocess', 'corpus_filtered.txt')
    
    if STAGE == 0:
        if args.max_len > 0 and args.data == 'FNS2022':
            for split in ['training']: # better not to edit test files
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
        for split in splits:
            for folder in ['annual_reports', 'gold_summaries']:
                for file_name in tqdm(os.listdir(os.path.join(DATASET_PATH, split, folder))):
                    tokenizer(os.path.join(DATASET_PATH, split, folder, file_name), os.path.join(DATASET_PATH, 'preprocess', split, folder, file_name), LANGUAGE, common_bow)
                print(f'{split} {folder} processed!')
        STAGE = 4
    
    if STAGE == 4 and os.path.exists(CORPUS_FILTERED_PATH):
        train_word2vec(DATASET_PATH, CORPUS_FILTERED_PATH, args.emb_dim)
        print('Word2Vec model saved!')
        STAGE = 5
    
    if STAGE == 5 and \
        len(os.listdir(os.path.join(DATASET_PATH, 'preprocess', 'training', 'annual_reports'))) > 0 and \
        len(os.listdir(os.path.join(DATASET_PATH, 'preprocess', 'training', 'gold_summaries'))) > 0 and \
        len(os.listdir(os.path.join(DATASET_PATH, 'preprocess', 'validation', 'annual_reports'))) > 0 and \
        len(os.listdir(os.path.join(DATASET_PATH, 'preprocess', 'validation', 'gold_summaries'))) > 0:
        total_time = time.time()
        total_labels = 0
        for split in splits:
            start_time = time.time()
            split, labels_number = label(DATASET_PATH, split, args.jit, TASK)
            elapsed_time = time.time() - start_time
            total_labels += labels_number
            print(f'Labels generated for the {split} set!')
            print(f'Elapsed time: {timedelta(elapsed_time)}')
            print(f'AVG time to process a label: {elapsed_time/labels_number} s')
        if args.data == 'FNS2022':
            split_data(os.path.join(DATASET_PATH, 'preprocess', 'labels'))
            print(f'Labels generated for the validation set!')
        print(f'Total elapsed time: {timedelta(time.time() - total_time)}')
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Run the preprocess pipeline.'
    )
    parser.add_argument('--data', type=str, default='FNS2022', choices={'FNS2022', 'CNN'}, help='Select the dataset.')
    parser.add_argument('--language', type=str, default='English', choices={'English', 'Greek', 'Spanish'}, help='Select the language if you use FNS2022.')
    parser.add_argument('--stage', type=int, default=0, choices={0, 1, 2, 3, 4, 5}, help=stage_help)
    parser.add_argument('--task', type=str, default='Headline Generation', choices={'Headline Generation', 'Summarization'}, help='Select the task to carry over the CNN dataset.')
    parser.add_argument('--emb_dim', type=int, default=300, action='store', help='The dimension of word embedding.')
    parser.add_argument('--max_len', type=int, default=1000, action='store', help='Limit the number of sentences in the articles for training purposes.')
    parser.add_argument('--jit', action="store_true", help='Optimize runtime performance using parallelization.')
    args = parser.parse_args()
    
    DATASET_PATH = '/content/NLP_Project/Dataset'
    splits = ['training', 'validation']
    if args.data == 'FNS2022':
        LANGUAGE = args.language
    else:
        LANGUAGE = 'English'
        TASK = args.task
        splits.append('test')
    DATASET_PATH = os.path.join(DATASET_PATH, args.data, LANGUAGE)
    if TASK is not None:
        DATASET_PATH = os.path.join(DATASET_PATH, TASK)
    
    if not os.path.exists(os.path.join(DATASET_PATH, 'preprocess')):
        for split in splits:
            os.makedirs(os.path.join(DATASET_PATH, 'preprocess', split, 'annual_reports'))
            os.makedirs(os.path.join(DATASET_PATH, 'preprocess', split, 'gold_summaries'))
        
    pipeline(DATASET_PATH, LANGUAGE, args.stage, splits)
