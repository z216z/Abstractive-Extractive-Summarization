import csv
import re

# Read files for the regex check:
with open('/content/NLP_Project/preprocess/regex/en_conversions.csv') as csvfile:
    en_conversions = [r for r in csv.reader(csvfile)]
with open('/content/NLP_Project/preprocess/regex/en_numerical_conversions.csv') as csvfile:
    en_numerical_conversions = [r for r in csv.reader(csvfile)]
with open('/content/NLP_Project/preprocess/regex/gr_conversions.csv') as csvfile:
    gr_conversions = [r for r in csv.reader(csvfile)]
with open('/content/NLP_Project/preprocess/regex/gr_numerical_conversions.csv') as csvfile:
    gr_numerical_conversions = [r for r in csv.reader(csvfile)]
with open('/content/NLP_Project/preprocess/regex/sp_conversions.csv') as csvfile:
    sp_conversions = [r for r in csv.reader(csvfile)]
with open('/content/NLP_Project/preprocess/regex/sp_numerical_conversions.csv') as csvfile:
    sp_numerical_conversions = [r for r in csv.reader(csvfile)]

def numerical_related_conversion(sentence, numerical_conversions):
    for row in numerical_conversions:
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
    return sentence
    
def abbreviation_conversion(sentence, conversions):
    for row in conversions:
        abbreviation = row[0]
        extended_version = row[1]
        sentence = re.sub(f'(^{abbreviation}\s)|(\s{abbreviation}\s)|(\s{abbreviation}$)',
                          " " + extended_version + " ", sentence)
    return sentence
    
def regex_check(sentence, language, use_abbreviations=True):
    if language == 'English':
        conversions = en_conversions
        numerical_conversions = en_numerical_conversions
    elif language == 'Greek':
        conversions = gr_conversions
        numerical_conversions = gr_numerical_conversions
    elif language == 'Spanish':
        conversions = sp_conversions
        numerical_conversions = sp_numerical_conversions
    else:
        raise ValueError('Error: language not available!')
    sentence = re.sub('[^a-zA-Z0-9$£€]', ' ', sentence)
    sentence = numerical_related_conversion(sentence, numerical_conversions)
    if use_abbreviations is False:
        sentence = abbreviation_conversion(sentence, conversions)
    sentence = re.sub(f'\s+', " ", sentence)
    sentence = re.sub(f'^\s|\s$', "", sentence)
    return sentence
