import nltk
from nltk.corpus import brown
from nltk.data import find
from google.cloud import bigquery
import numpy as np
import pandas as pd
import random
from gensim import models

#Set Google Cloud bucket domain
gcpd = 'w266-313114.final_project_clone'
# gcpd = 'w266-313317.final_project'
bqclient = bigquery.Client()



query = (
    """
    SELECT asset_id, STRING_AGG(lemma_string) as cn 
    FROM `{}.raw_tags` t1
    INNER JOIN `{}.lemmatized_tags` t2
    on t1.tag = t2.tag
    WHERE lemma_string != ''
    GROUP BY 1
    """.format(gcpd, gcpd)
)

result = bqclient.query(query).to_dataframe()

#split each tag into a separate column, keeping up to 50 tags per asset
expanded = result['cn'].str.split(',', 51, expand=True).replace('bffulltakes ftp', '')
expanded = expanded.drop([51], axis=1)

result['asset_id'] = 'assetnum' + result['asset_id'].astype(str)
embed_df = pd.concat([expanded, result.asset_id], axis=1)
lol_aid = embed_df.values.tolist()

#returns raw_tags
def retrieve_query():
    return(result)
#returns raw_tags with a separate column for each tag
def retrieve_expanded_query():
    return(expanded)

def retrieve_similar(asset_id_str):
    model = models.Word2Vec(lol_aid, vector_size=10, window=5, min_count=1, workers=4, compute_loss = True)
    return model.wv.most_similar(asset_id_str)

#returns w2v model
def retrieve_model(epochs, vec_size, window):
    return_model = models.Word2Vec(lol_aid, vector_size=vec_size, window=window, min_count=1, workers=4, compute_loss = True, epochs = epochs)
    return return_model
#returns w2v model on dataset with asset_id removed
def retrieve_model_no_id(epochs, vec_size, window):
    lol_lite = expanded.values.tolist()
    return_model_n = models.Word2Vec(lol_lite, vector_size=vec_size, window=window, min_count=1, workers=4, compute_loss = True, epochs = epochs)
    return return_model_n

#RARITY FUNCTIONS----------------------------------------------------------------------------------------------
rare_query = (
    """
    SELECT *
    FROM `{}.tag_asset_count_lemma`
    where rare_tag = 0
    """.format(gcpd)
)

rare_table = bqclient.query(rare_query).to_dataframe()
# rare_table = rare_table.astype(str)

##returns the table of non-rare items
def retrieve_rare_table():
    return rare_table

# def random_rare(num_tags):
#     for i in range(num_tags)

#returns True if word is not rare
def is_not_rare(checkword):
    for i in rare_table['tag']:
        if i == checkword:
            return True
        else:
            pass
    return False
 #this function changes rare tags to '[MASK]'
def mask_rare(string):
    if is_not_rare(string):
        string = string
    else:
            string = '[MASK]'
    return string

#this function deletes rare tags
def delete_rare(string):
    if is_not_rare(string):
        string = string
    else:
            string = ''
    return string

def retrieve_rare(num_rands):
    rand_list = []
    for i in range(num_rands):
        rand = random.randint(0,len(rare_table['tag']))
        rand_list.append(rare_table['tag'][rand])
    return rand_list
    
#LINEAR MODEL FUNCTIONS--------------------------------------------------------------------------------
lm_query = (
    """
    SELECT * 
    FROM `{}.reconstructed_assets`
    """.format(gcpd)
)

lm_result = bqclient.query(lm_query).to_dataframe()

#split each tag into a separate column, keeping up to 50 tags per asset
lm_expanded = lm_result['cn'].str.split(', ', 51, expand=True).replace('BFfulltakes_FTP', '')
lm_expanded = lm_expanded.drop([51], axis=1)
# lm_expanded = lm_expanded.replace(r"^ +| +$", r"", regex=True, inplace=True)x


lm_result['asset_id'] = 'assetnum' + lm_result['asset_id'].astype(str)
lm_embed_df = pd.concat([lm_expanded, lm_result.asset_id], axis=1)
lm_lol_aid = lm_embed_df.values.tolist()
    
def lm_retrieve_query():
    return(lm_result)

def lm_retrieve_expanded_query():
    return(lm_expanded)

def lm_retrieve_similar(asset_id_str):
    lm_model = models.Word2Vec(lol_aid, vector_size=10, window=5, min_count=1, workers=4, compute_loss = True)
    return lm_model.wv.most_similar(asset_id_str)

def lm_retrieve_model(epochs, vec_size, window):
    lm_return_model = models.Word2Vec(lm_lol_aid, vector_size=vec_size, window=window, min_count=1, workers=4, compute_loss = True, epochs = epochs)
    return lm_return_model

def lm_retrieve_model_na(epochs, vec_size, window):
    lm_lol_lite = lm_expanded.values.tolist()
    lm_return_model_na = models.Word2Vec(lm_lol_lite, vector_size=vec_size, window=window, min_count=1, workers=4, compute_loss = True, epochs = epochs)
    return lm_return_model_na

#lemma mapping-------------------------------------------------------------------------
lemma_query = (
    """
    SELECT * 
    FROM `{}.lemmatized_tags`
    """.format(gcpd)
)
lemma_table = bqclient.query(lemma_query).to_dataframe() 
def retrieve_lemma():
    return lemma_table

def lemma_map():
    lemma_dict = dict(zip(lemma_table.tag, lemma_table.lemma_string))
    return lemma_dict
    




