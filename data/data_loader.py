import pandas as pd
import numpy as np
import re
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
file_path= 'News Title.xls'
def data_load(file_path):
    df = pd.read_excel(file_path)
    title_column = 'News Title'
    category_column = 'Category'
    logger.info(f"Sample data from loaded file:\n{df.head(5)}")

    # Kiem tra cac gia tri bi thieu
    missing_titles = df[title_column].isna().sum()
    missing_categories = df[category_column].isna().sum()
    logger.info(f"Missing titles: {missing_titles}, Missing categories: {missing_categories}")

    # Xoa cac hang co gia tri bi thieu
    df = df.dropna(subset=[title_column, category_column])
    logger.info(f"After removing missing values: {len(df)} rows")

    titles = df[title_column]
    labels = df[category_column]

    # Thay the nhan bang so
    dict_labels = {'Entertainment': 0, 'Business': 1, 'Technology': 2, 'Medical': 3}
    processed_labels = []
    for label in labels:
        if label in dict_labels:
            processed_labels.append(dict_labels[label])
        else:
            logger.warning(f"Unknown category: {label}, assigning to default (0)")
            processed_labels.append(0)  # Default to first category

    return titles, np.array(processed_labels), dict_labels

def clean_data(titles):
    """Clean the news title data"""
    # Chuyen cac gia tri NaN thanh chuoi rong
    titles = titles.fillna('').astype(str)

    # Viet thuong cac tu
    titles = titles.apply(lambda x: x.lower())

    # Xoa cac dang viet tat
    titles = titles.apply(lambda x: clean_shortforms(x))

    # Xoa ky tu dac biet
    titles = titles.apply(lambda x: clean_symbol(x))

    return titles

def clean_shortforms(text):
    short_forms_dict = {
        "ain't": "is not", "aren't": "are not", "can't": "cannot",
        "'cause": "because", "could've": "could have", "couldn't": "could not",
        "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
        "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
        "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
        "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
        "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",
        "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
        "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
        "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
        "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
        "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
        "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
        "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
        "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
        "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
        "she'll've": "she will have", "she's": "she is", "should've": "should have",
        "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
        "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have",
        "that's": "that is", "there'd": "there would", "there'd've": "there would have",
        "there's": "there is", "here's": "here is", "they'd": "they would",
        "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
        "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
        "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
        "we'll've": "we will have", "we're": "we are", "we've": "we have",
        "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
        "what're": "what are", "what's": "what is", "what've": "what have",
        "when's": "when is", "when've": "when have", "where'd": "where did",
        "where's": "where is", "where've": "where have", "who'll": "who will",
        "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
        "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
        "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
        "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
        "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
        "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
        "you're": "you are", "you've": "you have"
    }

    clean_text = text
    for shortform in short_forms_dict.keys():
        if re.search(shortform, text):
            clean_text = re.sub(shortform, short_forms_dict[shortform], text)
    return clean_text

def clean_symbol(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
