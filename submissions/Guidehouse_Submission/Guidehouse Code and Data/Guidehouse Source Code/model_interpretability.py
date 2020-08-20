#-------------------------------------------------------------------------------
### packages
#-------------------------------------------------------------------------------

import os
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import nltk
nltk.download('wordnet')

#-------------------------------------------------------------------------------
### load data
#-------------------------------------------------------------------------------

def load_val():
    '''load validation dataset from repo'''
    fp_data = os.path.dirname(os.getcwd()) + '\\data\\raw\\'
    df = pd.read_csv(fp_data + 'AI_ML_Challenge_Training_Data_Set_v1.csv')
    df = df.rename(columns={'Clause ID': 'id',
                            'Clause Text': 'clause',
                            'Classification': 'class'})
    return df

def load_val_mac():
    '''load validation dataset from repo on a mac'''
    fp_data = os.path.dirname(os.getcwd()) + '/data/raw/'
    df = pd.read_csv(fp_data + 'AI_ML_Challenge_Validation_Data_Set_v1.csv')
    df = df.rename(columns={'Clause ID': 'id',
                            'Clause Text': 'clause',
                            'Classification': 'class'})
    return df


#-------------------------------------------------------------------------------
### clean clauses
#-------------------------------------------------------------------------------

def create_stopwords():
    '''create stopwords list for removal'''
    rmv = ['MoreWords']
    stopwords = nltk.corpus.stopwords.words('english') + rmv

    # apply same processing to stopwords as original clauses
    stopwords = [standardize_text(sw) for sw in stopwords]
    stopwords = set(stopwords)
    return stopwords


def remove_stopwords(cl):
    '''remove stopwords from term vector'''
    stopwords = create_stopwords()
    cl = [w for w in cl.split() if w not in stopwords]
    cl = ' '.join(cl)
    return cl


def standardize_text(cl):
    # lower
    cl = cl.lower()
    #remove section labels (e.g. 11.1, 1.2)
    cl = re.sub('\d+\.\d+', '', cl)
    #remove section labels pt 2 (e.g. 1., II.)
    cl = re.sub('^([0-9]+)|([IVXLCM]+)\\.?$', '', cl)
    #remove anything that isn't alphabetic or numeric character
    cl = re.sub('[^A-Za-z0-9]+', ' ', cl)
    #remove extra spaces
    cl = re.sub('\s+', ' ', cl)
    #remove leading/trailing whitespaces
    cl = cl.strip()
    return cl


def clean_clause(cl):
    cl = standardize_text(cl)
    cl = remove_stopwords(cl)
    return cl


def add_cln(cls):
    cln = [clean_clause(cl) for cl in cls]
    return cln


#-------------------------------------------------------------------------------
### read in data and return cleaned clauses with classes and ids
#-------------------------------------------------------------------------------

def process_val_clauses(df):
    
    # clean clauses
    df['clause_clean'] = add_cln(df['clause'])
    df = df[['id', 'clause_clean']]
    return df



#-------------------------------------------------------------------------------
### build voting classifier
#-------------------------------------------------------------------------------

def build_voting_classifier():
    lr = LogisticRegression(penalty = 'l1', C = 4.284709261868257, solver = 'saga', random_state = 123, )
    rf = RandomForestClassifier(
                                       criterion = 'gini',
                                       max_depth = 10,
                                       max_leaf_nodes = None,
                                       min_samples_leaf = 1,
                                       min_samples_split = 2,
                                       n_estimators = 100,
                                       random_state = 123)
    svc = SVC(C = 6.964691855978616, kernel = 'rbf', gamma = 'auto', max_iter = 10000, probability = True, random_state = 123)
    nb = GaussianNB()
    knn = KNeighborsClassifier(algorithm = 'kd_tree', metric = 'minkowski', n_neighbors = 3, p = 1, weights = 'distance')
    en = VotingClassifier(estimators = [('lr', lr), ('rf', rf), ('svc', svc), ('nb', nb), ('knn', knn)], 
                          voting = 'soft', 
                          weights = [1, 2, 1, 1, 1])
    return en

