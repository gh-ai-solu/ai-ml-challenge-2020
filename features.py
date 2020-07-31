#-------------------------------------------------------------------------------
### file description
#-------------------------------------------------------------------------------

### example code use ###
#df_trn = features.load_trn()
#df_features = features.gen_features(df_trn, 100)

#-------------------------------------------------------------------------------
### packages
#-------------------------------------------------------------------------------
import os
import pandas as pd
import string
import re

import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('words')
#nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

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
    cl = (re.compile('[%s]' % re.escape(chrs)).sub(' ', cl))
    cl = ' '.join(cl.split()) # extra whitespace
    cl = cl.strip() # leading and trailing whitespace
    cl = re.sub(r'\d+', ' ', cl) # remove numbers (not sure if this stays)
    return cl

### remove words ###

def create_stopwords():
    '''create stopwords list for removal'''
    rmv = ['MoreWords']
    stopwords = nltk.corpus.stopwords.words('english') + rmv

    # apply same processing to stopwords as original clauses
    stopwords = [clean_clause(sw) for sw in stopwords]
    return stopwords

def remove_stopwords(tv):
    '''remove stopwords from term vector'''
    stopwords = set(create_stopwords())
    tv = tv.copy()
    tv = [w for w in tv if w not in stopwords]    
    return tv

def remove_nonwords(tv, words):
    '''remove non-English words'''
    tv = tv.copy()
    tv = [w for w in tv if w in words]
    return tv

### stem term-vectors ###

def stem_tv(tv):
    '''stem a term vector'''
    tv = tv.copy() # keeps original column from being overwritten
    stemmer = PorterStemmer()
    tv = [stemmer.stem(w) for w in tv]
    return tv

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

### gen features ###

def gen_tfidf_fts(tvs):
    '''generate TF-IDF features from Series of term-vectors'''
    tfidf, v = tfidf_from_tvs(tvs)
    df_tfidf_fts = pd.DataFrame(tfidf.toarray(), columns=v.get_feature_names())
    return df_tfidf_fts

def gen_tfidf_pca_fts(tvs, ncomps):
    '''generate TF-IDF features then reduce via PCA'''
    tfidf, v = tfidf_from_tvs(tvs)
    df_tfidf_fts = pd.DataFrame(tfidf.toarray(), columns=v.get_feature_names())

    pca = PCA(n_components=ncomps)
    pcs = pca.fit_transform(df_tfidf_fts)

    colnames = ['pc{}'.format(i+1) for i in range(ncomps)]
    df_tfidf_pcs_fts = pd.DataFrame(pcs,
                              columns=colnames)
    return df_tfidf_pcs_fts

#-------------------------------------------------------------------------------
### raw data to features table
#-------------------------------------------------------------------------------
def gen_features(df, n_pcs):
    # copy to not overwrite
    df_preprocess = df.copy(deep=True)
    
    # clean clauses
    df_preprocess['clause_clean'] = (df_preprocess['clause']
                                       .apply(clean_clause))

    # remove unwanted words
    words = set(nltk.corpus.words.words())
    df_preprocess['clause_tv'] = (df_preprocess['clause_clean']
                                    .apply(nltk.word_tokenize))
    df_preprocess['clause_tv'] = (df_preprocess['clause_tv']
                                    .apply(remove_stopwords))
    df_preprocess['clause_tv'] = (df_preprocess['clause_tv']
                                    .apply(remove_nonwords, 
                                           args=(words,)))

    # stem
    df_preprocess['clause_tv_stemmed'] = (df_preprocess['clause_tv']
                                            .apply(stem_tv))

    # features
    df_tfidf_fts = gen_tfidf_fts(df_preprocess['clause_tv_stemmed'])
    df_tfidf_pcs_fts = gen_tfidf_pca_fts(df_preprocess['clause_tv_stemmed'], 
                                         n_pcs)

    # concat
    df_features = pd.concat([df_preprocess, 
                            df_tfidf_fts, 
                            df_tfidf_pcs_fts], 
                            axis=1)

    return df_features

def clean_features(df):
    '''function to fix data quality issues before modeling'''
    df['tv_len'] = (df['clause_tv_stemmed'].apply(len).value_counts())
    df = df[df['tv_len']>5] # stemmed tv more than 5 terms
    return

def df_csv_tempfolder(df, fname):
    fp_temp = os.path.dirname(os.getcwd()) + '\\ai-ml-challenge-2020\\temp\\'
    fp_csv =  fp_temp + fname + '.csv'
    df.to_csv(fp_csv)
    return

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
