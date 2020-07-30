#-------------------------------------------------------------------------------
### packages
#-------------------------------------------------------------------------------
import os
import pandas as pd
import string
import re

import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
nltk.download('punkt')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer

#-------------------------------------------------------------------------------
### load data
#-------------------------------------------------------------------------------
def load_trn():
    '''load training dataset from repo'''
    fp_data = os.path.dirname(os.getcwd()) + '\\ai-ml-challenge-2020\\data\\'
    df = pd.read_csv(fp_data + 'AI_ML_Challenge_Training_Data_Set_1_v1.csv')
    df = df.rename(columns={'Clause ID': 'id',
                            'Clause Text': 'clause',
                            'Classification': 'class'})
    return df

#-------------------------------------------------------------------------------
### pre-process data
#-------------------------------------------------------------------------------

### clean clauses ###

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

### remove stopwords ###

def create_stopwords():
    '''create stopwords list for removal'''
    rmv = ['MoreWords']
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

### convert to clauses to term-vectors ###

def clauses_to_tvs(cls):
    '''create term vector and remove stopwords'''
    tvs = [nltk.word_tokenize(cl) for cl in cls]
    tvs = [remove_stopwords(tv) for tv in tvs]
    return tvs

### stem term-vectors ###

def stem_tv(tv):
    '''stem a term vector'''
    tv = tv.copy() # keeps original column from being overwritten
    stemmer = PorterStemmer()
    for i in range(0, len(tv)):
        tv[i] = stemmer.stem(tv[i])
    return tv

def stem_tvs(tvs):
    tvs = [stem_tv(tv) for tv in tvs]
    return tvs

#-------------------------------------------------------------------------------
### Create TF-IDF Features
#-------------------------------------------------------------------------------    

### create TF-IDF ###

def tfidf_from_tvs(tvs):
    '''create TF-IDF from Series of term-vectors'''
    tvs = list(tvs.apply(lambda x: ' '.join(x)))
    v = TfidfVectorizer()
    tfidf = v.fit_transform(tvs)
    return tfidf, v

### gen TF-IDF features and append to input df ###

def features_from_tvs(tvs):
    '''generate TF-IDF features from Series of term-vectors'''
    tfidf, v = tfidf_from_tvs(tvs)
    df_features = pd.DataFrame(tfidf.toarray(), columns=v.get_feature_names())
    return df_features

def gen_tfidf_features(df, tv_col):
    '''gens TF-IDF features from df with a column of term-vectors and concats'''
    df_features = features_from_tvs(df[tv_col])
    df = pd.concat([df, df_features], axis=1)
    return df

#-------------------------------------------------------------------------------
### diagnostic functions
#-------------------------------------------------------------------------------
def print_cls_with_terms(df, clause_col, tv_col, terms, n):
    '''prints n clauses containing any of term-vector terms'''
    count = 0
    for i in range(len(df)):
        if count==n:
            break
        if bool(set(terms) & set(df[tv_col][i])):
            print('----------\n')
            print('clause index - {}\n'.format(i))
            print('{}'.format(df[clause_col][i]))
            count += 1

#-------------------------------------------------------------------------------
### example code
#-------------------------------------------------------------------------------
#import features

#df_trn = features.load_trn()

#df_preprocess = df_trn.copy(deep=True)
#df_preprocess['clause_clean'] = features.clean_clauses(df_preprocess['clause'])
#df_preprocess['clause_tv'] = features.clauses_to_tvs(df_preprocess['clause_clean'])
#df_preprocess['clause_tv_stemmed'] = features.stem_tvs(df_preprocess['clause_tv'])

#df_features = features.gen_tfidf_features(df_preprocess, 'clause_tv_stemmed')

#features.print_cls_with_terms(df_features, 
#                              'clause', 'clause_tv_stemmed', 
#                              ['aaa'], 10)
