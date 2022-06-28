import os
import sys
import json
import argparse
from tqdm import tqdm
from make_extraction_labels import label, split_data, analyze_documents
from plot_utils import plot_distributions
from preprocess_methods import *

stage_help = 'Select the starting analysis stage: \n' \
'0 - Generate Corpus \n' \
'1 - Generate BoW \n' \
'2 - Filter Corpus \n' \
'3 - Process Reports \n' \
'4 - Analyze Reports \n' \
'5 - Evaluate Distribution.'

def analyze_distribution(DATASET_PATH, LANGUAGE, STAGE):    
    CORPUS_TOKENIZED_PATH = os.path.join(DATASET_PATH, 'preprocess', 'distribution', 'corpus_tokenized.txt')
    CORPUS_FILTERED_PATH = os.path.join(DATASET_PATH, 'preprocess', 'distribution', 'corpus_filtered.txt')
    
    if STAGE == 0:
        generate_corpus(os.path.join(DATASET_PATH, 'training'), CORPUS_TOKENIZED_PATH, LANGUAGE)
        print('Corpus generated!')
        STAGE = 1
    
    if STAGE == 1 and os.path.exists(CORPUS_TOKENIZED_PATH):
        bow, common_bow = generate_bow(CORPUS_TOKENIZED_PATH)
        save_json(bow, os.path.join(DATASET_PATH, 'preprocess', 'distribution', 'bow.json'))
        save_json(common_bow, os.path.join(DATASET_PATH, 'preprocess', 'distribution', 'common_bow.json'))
        print('BoW generated!')
        STAGE = 2
    
    if STAGE == 2 and os.path.exists(CORPUS_TOKENIZED_PATH):
        try:
            common_bow = read_json(os.path.join(DATASET_PATH, 'preprocess', 'distribution', 'common_bow.json'))
        except ValueError:
            return False
        filter_corpus(CORPUS_TOKENIZED_PATH, CORPUS_FILTERED_PATH, common_bow)
        print('Corpus filtered!')
        STAGE = 3
    
    if STAGE == 3 and os.path.exists(CORPUS_TOKENIZED_PATH) and os.path.exists(CORPUS_FILTERED_PATH):
        try:
            common_bow = read_json(os.path.join(DATASET_PATH, 'preprocess', 'distribution', 'common_bow.json'))
        except ValueError:
            return False
        split = 'training'
        for folder in ['annual_reports', 'gold_summaries']:
            for file_name in tqdm(os.listdir(os.path.join(DATASET_PATH, split, folder))):
                tokenizer(os.path.join(DATASET_PATH, split, folder, file_name), os.path.join(DATASET_PATH, 'preprocess', 'distribution', split, folder, file_name), LANGUAGE, common_bow)
            print(f'{split} {folder} processed!')
        STAGE = 4
    
    if STAGE == 4 and len(os.listdir(os.path.join(DATASET_PATH, 'preprocess', 'distribution', 'training', 'annual_reports'))) > 0:
        analyze_documents(DATASET_PATH)
        print('Training documents analyzed!')
    
    if STAGE == 5 and len(os.listdir(os.path.join(DATASET_PATH, 'preprocess', 'distribution', 'analysis'))) > 0:
        plot_distributions(os.path.join(DATASET_PATH, 'preprocess', 'distribution', 'analysis'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Run the document distribution analysis.'
    )
    parser.add_argument('--data', type=str, default='FNS2022', choices={'FNS2022', 'CNN'}, help='Select the dataset.')
    parser.add_argument('--language', type=str, default='English', choices={'English', 'Greek', 'Spanish'}, help='Select the language if you use FNS2022.')
    parser.add_argument('--stage', type=int, default=0, choices={0, 1, 2, 3, 4, 5}, help=stage_help)
    args = parser.parse_args()
    
    DATASET_PATH = '/content/NLP_Project/Dataset'
    if args.data == 'FNS2022':
        LANGUAGE = args.language
    else:
        LANGUAGE = 'English'
    DATASET_PATH = os.path.join(DATASET_PATH, args.data, LANGUAGE)
    
    if not os.path.exists(os.path.join(DATASET_PATH, 'preprocess', 'distribution')):
        for split in ['training', 'validation']:
            os.makedirs(os.path.join(DATASET_PATH, 'preprocess', 'distribution',  split, 'annual_reports'))
            os.makedirs(os.path.join(DATASET_PATH, 'preprocess', 'distribution', split, 'gold_summaries'))
        
    analyze_distribution(DATASET_PATH, LANGUAGE, args.stage)
