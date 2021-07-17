import nltk
from nltk.corpus import brown
from nltk.data import find
from google.cloud import bigquery
import gensim
from gensim import models
import numpy as np
import pandas as pd

bqclient = bigquery.Client()
query = (
    """
    SELECT asset_id, STRING_AGG(lemma_string) as cn 
    FROM `w266-313114.final_project_clone.raw_tags` t1
    INNER JOIN `w266-313114.final_project_clone.lemmatized_tags` t2
    on t1.tag = t2.tag
    WHERE lemma_string != ''
    GROUP BY 1
    """
)
result = bqclient.query(query).to_dataframe()
    
def retrieve_query():
    return(result)


def retrieve_similar(asset_id_str):
    expanded = result['cn'].str.split(',', 20, expand=True)
    result['asset_id'] = 'assetnum' + result['asset_id'].astype(str)
    embed_df = pd.concat([expanded, result.asset_id], axis=1)
    lol_aid = embed_df.values.tolist()
    model = models.Word2Vec(lol_aid, vector_size=10, window=5, min_count=1, workers=4, compute_loss = True)
    
    return model.wv.most_similar(asset_id_str)


def retrieve_model(epochs, vec_size, window):
    expanded = result['cn'].str.split(',', 20, expand=True)
    result['asset_id'] = 'assetnum' + result['asset_id'].astype(str)
    embed_df = pd.concat([expanded, result.asset_id], axis=1)
    lol_aid = embed_df.values.tolist()
    model = models.Word2Vec(lol_aid, vector_size=vec_size, window=window, min_count=1, workers=4, compute_loss = True, epochs = epochs)
    
    return model