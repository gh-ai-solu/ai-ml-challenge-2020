import joblib
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

        data = input_data

        # Appending the embeddings per document
        EULA_docs = [data[i]['BERT_emdb'] for i in data]
        return EULA_docs

    def load_ensemble_model(self):
        """
        Grabs ensemble model from pickle and loads into class
        """
        # Grabs model from specified path
        en_model = joblib.load('../Guidehouse Compiled Models/en.joblib')
        return en_model

    def predict_clause(self):
        """
        Takes processed features and loaded model to predict outputs
        Returns the predictions as a list of lists
        """
        predictions = [self.en_model.predict(self.input_features[i]) for i in range(len(self.input_features))]
        return predictions

    def predict_probability(self):
        """
        Takes processed features and loaded model to produce prediction probability
        Returns the prediction probability as a list of lists
        """
        probas = [self.en_model.predict_proba(self.input_features[i]) for i in range(len(self.input_features))]
        return probas

    def metric_creation(self):
        """
        Returns updated F1 score of the ensemble model
        per document of clauses
        """
        # concat all EULA prediction lists together
        predictions_concat = np.concatenate(self.predictions)
        # Grab probabilities of positive class
        probas_pos = np.concatenate(self.probas)[:, 1]
        f1 = f1_score(np.array(self.labels), predictions_concat)
        brier = brier_score_loss(np.array(self.labels), probas_pos)

        return f1, brier

    def retrain_en_model(self):
        # STUB
        """
        Appends new text_data and labels to original training data and refits model
        """
        return "This will be an actual output"
