import argparse
import numpy as np

from src.data_handler import t3m_data_handler
from xgb_cv_classifier import xgbcvclassifier

#---------------------------------------------------------
#                   Input Arguments 
#---------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_card', type=str, help='Card containing all training parameters.', default='cards/xgb_card_2018UL_twoGlobalTracker.json')
parser.add_argument('-k', '--kFolds', type=int, help='Number of folds to be used for training.', default=1)
parser.add_argument('-t', '--tag', type=str, help='A tag (string) for all the files that are saved.', default='dh')
parser.add_argument('-w', '--path_to_weights', type=str, help='Path to the directory where weights will be stored.', default='xgb_weights/' )
parser.add_argument('-p' '--path_to_plots', type=str, help='Path to the directory where the training plots will be stored.', default='xgb_plots/')

args = parser.parse_args()

input_card = args.input_card
tag = args.tag
path_to_plots = args.path_to_plots
path_to_weights = args.path_to_weights
kfolds = args.kFolds
#---------------------------------------------------------


#---------------------------------------------------------
#                   import data handler
#---------------------------------------------------------
dh = t3m_data_handler()
dh.read_config_file(input_card)
dh.make_training_dataframes()
#---------------------------------------------------------


#---------------------------------------------------------
# train and test models for each category and cv folds
#---------------------------------------------------------

xgb_models = {}
multifold_cv = dh.make_cv_sets(kfolds)

for k in range(kfolds):
    for category_ in multifold_cv:

        X = multifold_cv[category_]['X']
        labels = multifold_cv[category_]['Y']
        weights = multifold_cv[category_]['weights']
        index = multifold_cv[category_]['index']

        train_set = (index!=k)
        valid_set = (index==k)

        train_x = X[train_set]
        train_y = labels[train_set]
        dh_x = X[valid_set]
        dh_y = labels[valid_set]
        train_weights = weights[train_set]

        model_name = 'had_tau3mu_cv_{}_category_{}_kFold_{}'.format(tag, category_, k+1)
        xgb_models[model_name] = xgbcvclassifier()
        
        with open(multifold_cv[category_]['settings'], 'rb') as f:
            settings = json.load(f)
        
        xgb_models[model_name].load_model_params(**settings)
        xgb_models[model_name].create_model()
        xgb_models[model_name].import_data({'train_x': train_x,
                        'train_y': train_y,
                        'weights': train_weights,
                        'test_x': test_x,
                        'test_y': test_y})

        xgb_models[model_name].train_model()
        xgb_models[model_name].evaluate_model(model_name, path_to_plots)
        xgb_models[model_name].save_model(model_name, path_to_weights)

average_preds = None

# get predictions

eval_set = dh.make_evaluation_set()
eval_cuts, eval_cutb = dh.get_eval_cuts()
xgb_scores = {}

for ch in eval_set:
    dataset = eval_set[ch]
    for category_ in multifold_cv:
        for ifold, k in enumerate(kfolds):
            # average predictions
            if ifold==0: average_preds = xgb_models['had_tau3mu_cv_{}_category_{}_kFold_{}'.format(tag, category_, k+1)].get_predictions()
            else: average_preds += xgb_models['had_tau3mu_cv_{}_category_{}_kFold_{}'.format(tag, category_, k+1)]

            # fold-wise predictions
            fold_preds = xgb_models['had_tau3mu_cv_{}_category_{}_kFold_{}'.format(tag, category_, k+1)].get_predictions()
        
        if ch=='BKG': average_preds = dataset.eval(eval_cutb[category_]).to_numpy()*average_preds
        else: average_preds = dataset.eval(eval_cuts[category_]).to_numpy()*average_preds
        average_preds = average_preds/kfolds;

# add the scores to the otuputfile
# name of the outputfile
path_to_output = dh.get_output_path()+'/'
path_to_output += dh.get_path_to_input().split('/')[-1].replace('.root', '_xgb_{}.root'.format(tag))

