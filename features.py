import os
import pandas as pd

def load_trn():
    fp_data = os.path.dirname(os.getcwd()) + '\\ai-ml-challenge-2020\\data\\'
    df = pd.read_csv(fp_data + 'AI_ML_Challenge_Training_Data_Set_1_v1.csv')
    return df
