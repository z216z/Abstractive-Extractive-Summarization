import os
import sys
import json
from preprocess_methods import tokenizer, generate_bow

"""
DATASET_PATH
FILTERED_CORPUS_PATH
FILTERED_REPORTS_PATH
FILTERED_SUMMARIES_PATH
BOW_PATH
"""

def pipeline(DATASET_PATH, FILTERED_CORPUS_PATH, FILTERED_REPORTS_PATH, FILTERED_SUMMARIES_PATH, BOW_PATH, language):
    
