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

class t3m_data_handler:
    """
    """
    def __init__(self):

        self.config = None
        self.input_file = ""
        self.output_path = ""
        
        self.trees = {}
        self.catrgoies = {}
        self.input_variables = {}
        self.spec_variables = {}
        self.settings = {}
        
        self.training_dataframes = {}
        
        self.bdt_cuts_train = {}
        self.bdt_cutb_train = {}

        self.bdt_cuts_eval = {}
        self.bdt_cutb_eval = {}

        self.year = ""
        self.lumi = ""

    def get_path_to_input(self):
        return self.input_file

    def get_output_path(self):
        return self.output_path

    def read_config_file(self, config_filename):

        with open(config_filename, 'rb') as file_:
            self.config = json.load(file_)
 
        # get the ntuple containing all trees
        if not ('ntuple' in self.config):
            print("Specify the input file!")
            return

        else:
            if not os.path.exists(self.config['ntuple']):
                print("Input file does not exist!")
                return

        self.input_file = self.config['ntuple']
        
        self.output_path = self.config['path_to_output']
        
        # get the list of trees (bkg, ds, bu, bd)
        if not 'trees' in self.config:
            print("Specify the list of trees!")
            return
        
        
        for tree_ in self.config['trees']:
            self.trees[tree_] = {
                                 'path': self.config['trees'][tree_]['path'],
                                 'weight':self.config['trees'][tree_]['weight'],
                                 'cuts': self.config['trees'][tree_]['cuts']
                                 }
        
        for category_ in self.config['mva']:
            self.bdt_cuts_train[category_] = self.config['mva'][category_]['cuts_train']
            self.bdt_cutb_train[category_] = self.config['mva'][category_]['cutb_train']
            self.bdt_cuts_eval[category_] = self.config['mva'][category_]['cuts_eval']
            self.bdt_cutb_eval[category_] = self.config['mva'][category_]['cutb_eval']
            self.input_variables[category_] = self.config['mva'][category_]['input_variables']
            self.spec_variables[category_] = self.config['mva'][category_]['spec_variables']
            self.settings[category_] = self.config['mva'][category_]['settings']
        
        # get luminosity and year
        self.year = self.config['year']

    def get_scale_factors(self, path_to_file='t3m_scale_factors.json'):
        # get scale factors
        with open(path_to_file, 'rb') as file_:
            sffile = json.load(file_)
        
        self.lumi = sffile[self.year]['Lumi']
        
        for tree_ in self.trees:
            self.scale_factors[tree_] = sffile[self.year]['SignalSF'][tree_]*sffile[self.year]['SignalNormSF']
    
    def get_input_variables(self, df, train_list, cuts=''):
        _df = None
        if cuts!='': _df = df[df.eval(cuts)]
        else: _df = df
        return _df[train_list].to_numpy()

    def get_indices(self, df, cuts=''):
        array = None
        if cuts!='': array = df.index.to_numpy()
        else: array = df[df.eval(cuts)].to_numpy()
        return array

    def load_data(self):
        '''
        filename: name of the input rootfile
        tree_dict: dictionary containing tree names and their correaponding paths
        varlist: list of variables to be extracted from the root file
        '''
        input_file = uproot.open(self.config['ntuple'])
        varlist = []
        for category_ in self.input_variables:
            for var_ in self.spec_variables[category_]+self.input_variables[category_]:
                if var_ not in varlist: varlist.append(var_)
            
        for tree_ in self.trees:
            self.trees[tree_]['dataframe'] = self.get_dataframe(input_file[self.trees[tree_]['path']], varlist)

    def get_dataframe(self, tree, branch_list):
        _dict = {}
        for _br in branch_list:
            _dict[_br] = getattr(tree[_br].arrays(), _br)
    
        return pd.DataFrame.from_dict(_dict)

    def get_eval_cuts(self):
        return self.bdt_cuts_eval, self.bdt_cutb_eval

    def make_evaluation_sets(self):
        
        training_variables = []

        for category_ in self.bdt_cuts_eval:
            tmpvariables = self.input_variables[category_]+spec_variables[category_]
            for var in tmpvariables:
                if var not in training_variables: training_variables.append(var)

        # make dataframes for each tree
        bkg_tree = self.trees['BKG']['dataframe']
        ds_tree = self.treees['DS']['dataframe']
        bu_tree = self.trees['BU']['dataframe']
        bd_tree = self.trees['BD]'['dataframe']

        bkg_trainX = self.get_input_variables(bkg_tree, training_variables, '')
        ds_trainX = self.get_input_variables(ds_tree, training_variables, '')
        bu_trainX = self.get_input_variables(bu_tree, training_variables, '')
        bd_trainX = self.get_input_variables(bd_tree, training_variables, '')

        eval_set = {'BKG': bkg_tree, 'DS': ds_tree, 'BU': but_tree, 'BD': bd_tree}

        return eval_set 


    def make_training_dataframes(self):

        # for a given category
        # implement signal cuts
        # implement background cuts
        
        self.load_data()

        for category_ in self.bdt_cuts_train:
            
            training_variables = self.input_variables[category_]
            
            # make dataframes for each tree
            bkg_tree = self.trees['BKG']['dataframe']
            ds_tree = self.trees['DS']['dataframe']
            bu_tree = self.trees['BU']['dataframe']
            bd_tree = self.trees['BD']['dataframe']
            
            bkg_trainX = self.get_input_variables(bkg_tree, training_variables, self.config['mva'][category_]['cutb_train'])
            ds_trainX = self.get_input_variables(ds_tree, training_variables, self.config['mva'][category_]['cuts_train'])
            bu_trainX = self.get_input_variables(bu_tree, training_variables, self.config['mva'][category_]['cuts_train'])
            bd_trainX = self.get_input_variables(bd_tree, training_variables, self.config['mva'][category_]['cuts_train'])

            bkg_index = self.get_indices(bkg_tree, self.config['mva'][category_]['cutb_train'])
            ds_index = self.get_indices(ds_tree, self.config['mva'][category_]['cuts_train'])
            bu_index = self.get_indices(bu_tree, self.config['mva'][category_]['cuts_train'])
            bd_index = self.get_indices(bd_tree, self.config['mva'][category_]['cuts_train'])
            index = np.concatenate((bkg_index, ds_index, bu_index, bd_index)) 

            bkg_trainY = np.zeros(len(bkg_trainX))
            ds_trainY = np.ones(len(ds_trainX))
            bu_trainY = np.ones(len(bu_trainX))
            bd_trainY = np.ones(len(bd_trainX))

            bkg_weights = np.ones(len(bkg_trainX))
            ds_weights = ds_trainY*0.72
            bu_weights = bu_trainY*0.14
            bd_weights = bd_trainY*0.14
            
            # define index for cross validation

            # combine signal and baground into one array
            X = np.concatenate((bkg_trainX, ds_trainX, bu_trainX, bd_trainX))
            Y = np.concatenate((bkg_trainY, ds_trainY, bu_trainY, bd_trainY))
            weights = np.concatenate((bkg_weights, ds_weights, bu_weights ,bd_weights))
            
            self.training_dataframes[category_] = {'index': index,
                                                   'X': X,
                                                   'Y': Y,
                                                   'weights': weights}

    def make_cv_sets(self, kFolds=5):
        # make cv index
        if kFolds<1: 
            print('kFold needs to be an integer larger than 1')
            return {}
        cv_df = {}
        for category_ in self.training_dataframes:
            cv_df[category_] = self.training_dataframes[category_].copy()
            cv_df[category_]['index'] = cv_df[category_]['index']%kFolds
        
        return cv_df
