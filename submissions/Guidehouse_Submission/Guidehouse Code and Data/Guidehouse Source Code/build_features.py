#-------------------------------------------------------------------------------
### packages
#-------------------------------------------------------------------------------
import json
import re
import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer 
import numpy as np
import torch
import transformers as ppb # pytorch transformers
nltk.download('wordnet')

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
    #remove tabs
    cl = re.sub('\t', ' ', cl)
    #remove linebreak
    cl = re.sub('\n', ' ', cl)
    #remove leading/trailing whitespaces
    cl = cl.strip()
    return cl


def clean_clause(cl):
    cl = standardize_text(cl)
    cl = remove_stopwords(cl)
    return cl


def add_cln(raw):
    cln = raw.copy()
    for doc,cls in raw.items():
        cln[doc] = {
            'raw': cls,
            'cln': [clean_clause(cl) for cl in cls]
        }
    return cln


#-------------------------------------------------------------------------------
### stem and lem clauses
#-------------------------------------------------------------------------------
def lemm_cl(cl):
    lemmer = WordNetLemmatizer()
    cl = [lemmer.lemmatize(w) for w in cl.split()]
    cl = ' '.join(cl)
    return cl


def stem_cl(cl):
    stemmer = PorterStemmer()
    cl = [stemmer.stem(w) for w in cl.split()]
    cl = ' '.join(cl)
    return cl


def add_stm(cln):
    stm = cln.copy()
    for k in stm.keys():
        stm[k]['stm'] = [stem_cl(cl) for cl in stm[k]['cln']]
    return stm


def add_lem(cln):
    lem = cln.copy()
    for k in lem.keys():
        lem[k]['lem'] = [lemm_cl(cl) for cl in lem[k]['cln']]
    return lem


#-------------------------------------------------------------------------------
### BERT tokens and embeddings
#-------------------------------------------------------------------------------
def BERT_tkns(cls, tokenizer):
    tkns_cls = []
    for cl in cls:
        # print(cl)
        tkns_cls.append(tokenizer.encode(cl, add_special_tokens=True))
    # tkns_cls = [tokenizer.encode(cl, add_special_tokens=True) for cl in cls]
    for i in range(0, len(tkns_cls)):
        if len(tkns_cls[i]) > 512:
            tkns_cls[i] = tkns_cls[i][:512] # only keep first 512 tokens due to BERT limit
        tkns_cls[i] = np.array(tkns_cls[i] + [0]*(512-len(tkns_cls[i])))
    return tkns_cls


def BERT_attn(tkns_cls):
    attn_cls = [np.where(tkns != 0, 1, 0) for tkns in tkns_cls]
    return attn_cls


def add_BERT(cln):
    BERT = cln.copy()

    model_class, tokenizer_class, pretrained_weights = (
        ppb.DistilBertModel,
        ppb.DistilBertTokenizer,
        'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    for k in BERT.keys():
        BERT[k]['BERT_tkns'] = BERT_tkns(BERT[k]['cln'], tokenizer)
        BERT[k]['BERT_attn'] = BERT_attn(BERT[k]['BERT_tkns'])
        tkns = np.array(BERT[k]['BERT_tkns'])
        if len(BERT[k]['BERT_tkns']) > 1:
            tkns = np.squeeze(tkns)
        tkns = torch.tensor(tkns, dtype=torch.long)

        attn = np.array(BERT[k]['BERT_attn'])
        if len(BERT[k]['BERT_attn']) > 1:
            attn = np.squeeze(attn)
        attn = torch.tensor(attn)
        with torch.no_grad():
            embds = model(tkns, attention_mask=attn)

        BERT[k]['BERT_embd'] = embds[0][:, 0, :].numpy()
    return BERT


#-------------------------------------------------------------------------------
### create full feature set
#-------------------------------------------------------------------------------
def load_json(fp):
    with open(fp, 'r') as f:
        raw = json.load(f)
    return raw

# example load
# raw = load_json("C:/repos/EULA/temp/sample.json")
# raw1 = next(iter(raw.items()))
# raw1 = {raw1[0]:raw1[1]}

# rawtrn = pd.read_csv(r"C:\repos\EULA\data\raw\AI_ML_Challenge_Training_Data_Set_1_v1.csv")
# rawtrn = {str(x):[y] for x,y in zip(np.array(rawtrn['Clause ID']),np.array(rawtrn['Clause Text']))}


def keep_fts(fts):
    fts_k = {}
    for doc, cls in fts.items():
        fts_k[doc] = {
            'BERT_emdb': fts[doc]['BERT_embd']
        }
    return fts_k


def gen_fts(raw):
    '''
    create features from JSON

    output structures as dict -
    {doc_name: {
        'raw': list of clauses with raw text from original JSON
        'cln': list of clauses with cleaned text
        'stm': list of clauses stemmed
        'lem': list of clauses
        'BERT_tkns': BERT tkns padded to length 512
        'BERT_attn': BERT attention mask at length 512
        'BERT_embd': BERT embeddings
    }}
    '''
    fts = add_cln(raw)  # convert raw JSON to dict and apply cleaning
    # fts = add_stm(fts)  # add stemmed clause
    # fts = add_lem(fts)  # add lemmed clause
    fts = add_BERT(fts)  # add BERT tokens, attention mask, and

    fts_k = keep_fts(fts)
    return fts_k

# exploring output - output for first clause in first document
# firstk = next(iter(fts))
# print(fts[firstk]['raw'][0] + "\n")
# print(fts[firstk]['cln'][0] + "\n")
# print(fts[firstk]['stm'][0] + "\n")
# print(fts[firstk]['lem'][0] + "\n")
# print(fts[firstk]['BERT_tkns'][0])
# print(fts[firstk]['BERT_attn'][0])
# print(fts[firstk]['BERT_embd'][0])
