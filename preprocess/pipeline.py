import os
import sys
import json
import argparse

from preprocess_methods import generate_corpus, tokenizer, generate_bow, filter_corpus, save_json, read_json
    
def pipeline(DATASET_PATH):    
    CORPUS_TOKENIZED_PATH = os.path.join(DATASET_PATH, 'preprocess', 'corpus_tokenized.txt')
    CORPUS_FILTERED_PATH = os.path.join(DATASET_PATH, 'preprocess', 'corpus_filtered.txt')
    
    generate_corpus(os.path.join(DATASET_PATH, 'training'), CORPUS_TOKENIZED_PATH)
    print('Corpus generated!')
    
    bow, common_bow = generate_bow(CORPUS_TOKENIZED_PATH)
    save_json(bow, os.path.join(DATASET_PATH, 'preprocess', 'bow.json'))
    save_json(common_bow, os.path.join(DATASET_PATH, 'preprocess', 'common_bow.json'))
    print('BoW generated!')
    
    filter_corpus(CORPUS_TOKENIZED_PATH, CORPUS_FILTERED_PATH, common_bow)
    print('Corpus filtered!')
    
    for _, folder in enumerate(os.listdir(os.path.join(DATASET_PATH, 'training'))): # folders = [annual_reports, golden_summaries]
        for i, file_name in enumerate(os.listdir(os.path.join(DATASET_PATH, 'training', folder))):
            tokenizer(os.path.join(DATASET_PATH, 'training', folder, file_name), os.path.join(DATASET_PATH, 'preprocess', folder, file_name), common_bow)
        print(f'{folder} processed!')
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Run the preprocess pipeline.'
    )
    parser.add_argument('--data', type=str, default='FNS2022', choices={'FNS2022', 'CNN'}, help='Select the dataset.')
    parser.add_argument('--language', type=str, default='English', choices={'English', 'Greek', 'Spanish'}, help='Select the language.')
    args = parser.parse_args()
    
    DATASET_PATH = '/content/NLP_Project/Dataset'
    if args.data == 'FNS2022':
        DATASET_PATH = os.path.join(DATASET_PATH, args.data, args.language)
    else:
        DATASET_PATH = os.path.join(DATASET_PATH, args.data)
    
    if not os.path.exists(os.path.join(DATASET_PATH, 'preprocess')):
        os.makedirs(os.path.join(DATASET_PATH, 'preprocess', 'annual_reports'))
        os.makedirs(os.path.join(DATASET_PATH, 'preprocess', 'gold_summaries'))
        
    pipeline(DATASET_PATH)
