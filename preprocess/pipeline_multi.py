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
'1 - Process Reports and Summaries \n' \
'2 - Generate Labels.'

def pipeline(DATASET_PATH, STAGE):    
    CORPUS_PATH = os.path.join(DATASET_PATH, 'Multi', 'preprocess', 'corpus.txt')
    CORPUS_FILTERED_PATH = os.path.join(DATASET_PATH, 'Multi', 'preprocess', 'corpus_filtered.txt')
    
    if STAGE == 0:
        if args.max_len is not None:
            for lang in ['English', 'Greek', 'Spanish']:
                for _, split in enumerate(['training', 'validation']):
                    for i, file_name in enumerate(os.listdir(os.path.join(DATASET_PATH, lang, split, 'annual_reports'))):
                        cut_document(os.path.join(DATASET_PATH, lang, split, 'annual_reports', file_name), args.max_len)
                generate_corpus(os.path.join(DATASET_PATH, lang, 'training'), CORPUS_PATH, lang, mode='a')
        print('Corpus generated!')
        STAGE = 1
    
    if STAGE == 1 and os.path.exists(CORPUS_TOKENIZED_PATH) and os.path.exists(CORPUS_FILTERED_PATH):
        for lang in ['English', 'Greek', 'Spanish']:
            for split in ['training', 'validation']:
                for folder in ['annual_reports', 'gold_summaries']:
                    for file_name in tqdm(os.listdir(os.path.join(DATASET_PATH, split, folder))):
                        if not os.path.exists(DATASET_PATH, 'Multi', 'preprocess', lang, split):
                            os.makedirs(os.path.join(DATASET_PATH, 'Multi', 'preprocess', lang, split))
                        tokenizer(os.path.join(DATASET_PATH, lang, split, folder, file_name), os.path.join(DATASET_PATH, 'Multi', 'preprocess', lang, split, folder, file_name), lang, None)
                print(f'{split} {folder} processed!')
        STAGE = 2
    
    if STAGE == 2 and \
        len(os.listdir(os.path.join(DATASET_PATH, 'Multi', 'preprocess'))) > 0:
        for lang in ['English', 'Greek', 'Spanish']:
            for split in ['training', 'validation']:
                label_multi(DATASET_PATH, lang, split)
                split = 'train' if split == 'training' else 'test'
                print(f'Labels generated for the {split} set!')
            split_data(os.path.join(DATASET_PATH, 'Multi', 'preprocess', lang, 'labels'))
        print(f'Labels generated for the validation set!')
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Run the preprocess pipeline.'
    )
    parser.add_argument('--stage', type=int, default=0, choices={0, 1, 2}, help=stage_help)
    parser.add_argument('--emb_dim', type=int, default=300, action='store', help='The dimension of word embedding.')
    parser.add_argument('--max_len', type=int, default=1000, action='store', help='Limit the number of sentences in the articles for training purposes.')
    args = parser.parse_args()
    
    DATASET_PATH = '/content/NLP_Project/Dataset/FNS2022'
    
    if not os.path.exists(os.path.join(DATASET_PATH, 'Multi')):
        os.makedirs(os.path.join(DATASET_PATH, 'Multi', 'preprocess'))
        
    pipeline(DATASET_PATH, args.stage)
