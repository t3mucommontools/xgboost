import os, sys
import json
from datetime import time

import numpy as np
import pandas as pd
import awkward as ak
import uproot

import json
import pickle

import xgboost
from xgboost import XGBClassifier

# import sklearn packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import label_binarize, LabelEncoder

from matplotlib import pyplot as plt

class xgbcvclassifier:

    def __init__(self):

        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.train_weights = None
        self.test_weights = None
        self.model = None
        self.model_params = {}
        self.model_history = None

    def load_model(self, filepath):
        """
        Load an existing model using pickle file
        """
        self.model = load_model(filepath)

    def create_model(self, settings='cards/default_settings.json', **kwargs):
        """
        Create model and load parameters from a json file or
        load them as keywords arguments
        """
        self.model = xgboost.XGBRegressor()
        if paramfile and os.path.exists(settings):
            with open(settings, 'rb') as file_:
                self.model_params = json.load(file_)

        for s in kwargs:
            try:
                self.model_params[s] = kwargs[s]
            except(KeyError):
                print('%s if not a parameter in XGBRegressor!' % s)
        
        # set parameters
        for arg in self.model_params:
            setattr(self.model, arg, self.model_params[arg])

    def train_model(self, eval_metric="error"):

        # Specify which dataset and which metric should be used for early stopping.
        #early_stop = EarlyStopping(metric_name='error')

        self.model.fit(self.train_x, self.train_y,
                eval_set = [(self.train_x, self.train_y), (self.test_x, self.test_y)],
                eval_metric=eval_metric,
                #callbacks=[early_stop],
                sample_weight=self.train_weights)

    def get_predictions(self, X):
        return self.model.predict(X)

    def evaluate_model(self, tag='test', path_to_output='./'):
        results = self.model.evals_result()
        epochs = len(results['validation_0']['error'])
        x_axis = range(0, epochs)
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['error'], label='Train')
        ax.plot(x_axis, results['validation_1']['error'], label='Test')
        ax.legend()
        plt.ylabel('Error')
        plt.title('XGBoost Error')
        plt.savefig(path_to_output+'/xgbclassifier_classifier_{}.png'.format(tag))

    def import_data(self, df_dict):
        self.train_x = df_dict['train_x']
        self.train_y = df_dict['train_y']
        self.weights = df_dict['weights']
        self.test_x = df_dict['test_x']
        self.test_y = df_dict['test_y']

    def save_model(self, tag='test', path_to_output='./'):
        pickle.dump(self.model, open(path_to_output+'/xgbclassifier_'+tag+'.pkl','wb'))
