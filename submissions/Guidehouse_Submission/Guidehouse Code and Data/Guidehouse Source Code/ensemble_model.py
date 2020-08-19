# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy.stats import randint, uniform
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import joblib
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
from sklearn.metrics import f1_score, brier_score_loss
import pickle


class EnsembleClass():
    """
    training_data = "BERT_features.csv"
    """
    def __init__(self, input_data, labels=None, training_data=None):
        self.input_features = self.process_input_features(input_data)
        self.en_model = self.load_ensemble_model()
        self.labels = labels
        self.training_data = training_data

    def process_input_features(self, input_data):
        """
        Takes featurization team's input and converts it into
        pd dataframe for our ensemble model to use
        """
        # with open(input_data, 'rb') as fp:
        #     # This next line will be depricated once the feature team's code
        #     # passes in their features as the input to this function
        #     # This will eventually be an input JSON/python dictionary
        #     data = pickle.load(fp)

        #     # This next line will eventually be uncommented once input_data is
        #     # the data the models need to use

        data = input_data

        # Appending the embeddings per document
        EULA_docs = []
        for i in data:
            EULA_docs.append(data[i]['BERT_emdb'])
        return EULA_docs

    def load_ensemble_model(self):
        """
        Grabs ensemble model from pickle and loads into class
        """
        # Grabs model from specified path
        en_model = joblib.load('en.joblib')
        return en_model

    def predict_clause(self):
        """
        Takes processed features and loaded model to predict outputs
        Returns the predictions as a list of lists
        """
        predictions = []

        # Appending predictions per document into a list of lists
        for i in range(len(self.input_features)):
            predictions.append(self.en_model.predict(self.input_features[i]))
        return predictions

    def predict_probability(self):
        """
        Takes processed features and loaded model to produce prediction probability
        Returns the prediction probability as a list of lists
        """
        proba = []

        # Appending probability per document into a list of lists
        for i in range(len(self.input_features)):
            proba.append(self.en_model.predict_proba(self.input_features[i]))
        return proba

    def metric_creation(self):
        """
        Returns updated F1 score of the ensemble model
        per document of clauses
        """
        f1 = []
        # Appending F1 scores per document into a list of scores
        for i in range(len(self.input_features)):
            f1.append(f1_score(self.labels[i], self.predictions[i], average=None))
        return f1

    def retrain_en_model(self):
        # STUB
        """
        Appends new text_data and labels to original training data and refits model
        """
        return "This will be an actual output"
