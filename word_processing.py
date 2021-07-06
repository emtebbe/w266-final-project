import pandas as pd
from google.cloud import bigquery
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
import string
from nltk.corpus import stopwords

import tensorflow as tf
from tensorflow import keras
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

bqclient = bigquery.Client()

def word_preprocessing(word):
    lower = word.lower()
    punct_replacer = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    rem_punct = lower.translate(punct_replacer)
    lemma = [lemmatizer.lemmatize(w) for w in nltk.word_tokenize(rem_punct)]
    rem_stop = [w for w in lemma if not w in stop_words]
    rem_digits = [re.sub('\d', '<dig>', i) for i in rem_stop]
    return rem_digits, word

def preprocess_tags():
    query = (
        """
        SELECT * FROM `w266-313317.final_project.tag_asset_count`
        WHERE language = 'en'
        """
    )
    result = bqclient.query(query).to_dataframe()
    
    lemmatizer = WordNetLemmatizer()
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    stop_words = set(stopwords.words('english'))
    
    result_np = result[['tag','language','auto_generated','organization_id','asset_count']].to_numpy()
    lemmatized = [word_preprocessing(i[0]) for i in result_np]
    lemma_df = pd.DataFrame(lemmatized, columns = ['lemmatized','tag']).set_index('tag')
    
    query = (
        """
        SELECT tag, cast(asset_id as string) as asset_id, 
        auto_generated, 
        cast(organization_id as string) as organization_id, 
        source
        FROM `w266-313317.final_project.raw_tags`
        WHERE language = 'en'
        """
    )
    full_tags = bqclient.query(query).to_dataframe()
    
    
