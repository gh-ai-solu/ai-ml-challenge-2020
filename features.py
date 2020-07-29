import os
import pandas as pd
import string
import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')

def load_trn():
    '''load training dataset from repo'''
    fp_data = os.path.dirname(os.getcwd()) + '\\ai-ml-challenge-2020\\data\\'
    df = pd.read_csv(fp_data + 'AI_ML_Challenge_Training_Data_Set_1_v1.csv')
    df = df.rename(columns={'Clause ID': 'id',
                            'Clause Text': 'clause',
                            'Classification': 'class'})
    return df

def clean_clause(cl):
    '''clean clauses before further processing'''
    # lower case
    cl = cl.lower()
    
    # remove punctuation and other characters
    other_chars = ('“', '”', '’', '‘', '\\')
    chrs = string.punctuation.join(other_chars)
    cl = (re.compile('[%s]' % re.escape(chrs)).sub('', cl))
    cl = " ".join(cl.split()) # extra whitespace
    cl = cl.strip() # leading and trailing whitespace
    cl = re.sub(r'\d+', '', cl) # remove numbers (not sure if this stays)
    return cl

def clean_clauses(cls):
    cls = cls.apply(clean_clause)
    return cls

def create_stopwords():
    '''create stopwords list for removal'''
    rmv = ['customer']
    stopwords = nltk.corpus.stopwords.words('english') + rmv

    # apply same processing to stopwords as original clauses
    stopwords = [clean_clause(sw) for sw in stopwords]
    return stopwords

def remove_stopwords(tv):
    '''remove stopwords from term vector'''
    stopwords = create_stopwords()
    newtv = []
    for t in tv:
        if t not in stopwords:
            newtv.append(t)    
    return newtv

def clauses_to_tvs(cls):
    '''create term vector and remove stopwords'''
    tvs = [nltk.word_tokenize(cl) for cl in cls]
    tvs = [remove_stopwords(tv) for tv in tvs]
    return tvs

## example code-use
#df_trn = features.load_trn()
#df_trn['clause'] = features.clean_clauses(df_trn['clause'])
#df_trn['clause_tvs'] = features.clauses_to_tvs(df_trn['clause'])
