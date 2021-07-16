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
    stripped = lower.strip()
    punct_replacer = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    rem_punct = stripped.translate(punct_replacer)
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
    result['rare_tag'] = result['asset_count'] <= 3
    result['rare_tag'] = result['rare_tag'].astype(int)
    
    lemmatizer = WordNetLemmatizer()
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    stop_words = set(stopwords.words('english'))
    
    result_np = result[['tag','language','auto_generated','organization_id','asset_count']].to_numpy()
    lemmatized = [word_preprocessing(i[0]) for i in result_np]
    lemma_df = pd.DataFrame(lemmatized, columns = ['lemmatized','tag']).set_index('tag')
    lemma_df['lemma_string'] =  [" ".join(map(str, l)) for l in lemma_df['lemmatized']]
    lemma_bq_df = lemma_df[['tag', 'lemma_string']].drop_duplicates()
    lemma_bq_df = lemma_bq_df[lemma_bq_df.lemma_string.notnull() & lemma_bq_df.lemma_string.notna()]
    join_lemma = lemma_bq_df.set_index('tag')
    with_lemma = result.join(join_lemma, on='tag')
    job_config = bigquery.LoadJobConfig(
        # Specify a (partial) schema. All columns are always written to the
        # table. The schema is used to assist in data type definitions.
        schema=[
            # Specify the type of columns whose type cannot be auto-detected. For
            # example the "title" column uses pandas dtype "object", so its
            # data type is ambiguous.
            bigquery.SchemaField("tag", bigquery.enums.SqlTypeNames.STRING),
            # Indexes are written if included in the schema by name.
            bigquery.SchemaField("lemma_string", bigquery.enums.SqlTypeNames.STRING),
        ],
        # Optionally, set the write disposition. BigQuery appends loaded rows
        # to an existing table by default, but with WRITE_TRUNCATE write
        # disposition it replaces the table with the loaded data.
        write_disposition="WRITE_TRUNCATE",
    )
    table_id = "w266-313317.final_project.tag_asset_count_lemma"
    job = bqclient.load_table_from_dataframe(
        with_lemma, table_id, job_config=job_config
    )  # Make an API request.
    job.result()
    print("table uploaded to bigquery at w266-313317.final_project.tag_asset_count_lemma")
    
    
