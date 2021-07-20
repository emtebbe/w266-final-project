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
    SELECT * 
    FROM `w266-313114.final_project_clone.reconstructed_assets`
    """
)
result = bqclient.query(query).to_dataframe()
expanded = result['cn'].str.split(',', 50, expand=True)
result['asset_id'] = 'assetnum' + result['asset_id'].astype(str)
embed_df = pd.concat([expanded, result.asset_id], axis=1)
lol_aid = embed_df.values.tolist()
    
def retrieve_query():
    return(result)
def retrieve_expanded_query():
    return(expanded)

def retrieve_similar(asset_id_str):
    model = models.Word2Vec(lol_aid, vector_size=10, window=5, min_count=1, workers=4, compute_loss = True)
    return model.wv.most_similar(asset_id_str)

def retrieve_model(epochs, vec_size, window):
    return_model = models.Word2Vec(lol_aid, vector_size=vec_size, window=window, min_count=1, workers=4, compute_loss = True, epochs = epochs)
    return return_model

def retrieve_model_na(epochs, vec_size, window):
    lol_lite = expanded.values.tolist()
    return_model_na = models.Word2Vec(lol_lite, vector_size=vec_size, window=window, min_count=1, workers=4, compute_loss = True, epochs = epochs)
    return return_model_na