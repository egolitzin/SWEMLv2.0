# This file created on 02/20/2024 by savalan

import numpy as np
from hydrotools.nwm_client import utils 
import xgboost as xgb
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import pickle as pkl
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import optuna
from matplotlib import pyplot
import pyarrow as pa
import pyarrow.parquet as pq

# deep learning packages
import torch


import os
import sys
import warnings
sys.path.insert(0, '../..')
from model_scripts import Simple_Eval
warnings.filterwarnings("ignore")

HOME = os.getcwd()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

class XGBoostRegressorCV:
    def __init__(self, params, path=None):
        self.params = params
        self.model = xgb.XGBRegressor(objective=self.params['objective'])
        #self.model = xgb.XGBRegressor(tree_method="hist", device="cuda"
        self.best_model = None
        self.path = path

        if DEVICE =='cuda':
            self.model = xgb.XGBRegressor(tree_method='gpu_hist', objective=self.params['objective'])
            print(f"XGBoost model using GPU")


    def tune_gridsearch(self, X, y, cv=3):
        """Performs GridSearchCV to find the best hyperparameters."""
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.params,
            scoring='neg_root_mean_squared_error',
            cv=cv,
            n_jobs=-1,
            verbose=3
        )
        if DEVICE == 'cuda':
            grid_search.fit(X, y, tree_method='gpu_hist')
        else:
            grid_search.fit(X, y)
        self.best_model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best RMSE: {grid_search.best_score_}")
        pkl.dump(grid_search, open(self.path, "wb"))
        return grid_search.best_params_

    def tune_hyperparameters(self, X, y, cv=3, n_trials=50):
        """Uses Optuna TPE Bayesian optimization to find best hyperparameters.

        Hyperparameter guidance for SWE/snow modeling:
          max_depth: deeper trees (8-10) capture complex elevation-precip interactions
            but risk overfitting across basins; shallower (3-6) generalizes better
          n_estimators: more trees with low eta improves accuracy; early stopping
            (50 rounds) prevents overfitting so upper bound is less critical
          eta (learning rate): 0.1 trains faster; 0.01-0.05 often improves final
            RMSE but requires proportionally more estimators
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        meta_keys = {'objective', 'random_state', 'n_jobs'}

        objective_val = self.params.get('objective', ['reg:squarederror'])
        if isinstance(objective_val, list):
            objective_val = objective_val[0]

        random_state = self.params.get('random_state', [42])
        if isinstance(random_state, list):
            random_state = random_state[0]

        n_jobs = self.params.get('n_jobs', [-1])
        if isinstance(n_jobs, list):
            n_jobs = n_jobs[0]

        fixed = {}
        tunable = {}
        for key, values in self.params.items():
            if key in meta_keys:
                continue
            vals = values if isinstance(values, list) else [values]
            if len(vals) == 1:
                fixed[key] = vals[0]
            else:
                tunable[key] = vals

        tree_method = 'gpu_hist' if DEVICE == 'cuda' else 'hist'
        X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=random_state)

        def optuna_objective(trial):
            params = {k: trial.suggest_categorical(k, v) for k, v in tunable.items()}
            params.update(fixed)
            model = xgb.XGBRegressor(
                objective=objective_val,
                random_state=random_state,
                n_jobs=n_jobs,
                tree_method=tree_method,
                eval_metric='rmse',
                early_stopping_rounds=50,
                **params
            )
            model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            preds = model.predict(X_v)
            return mean_squared_error(y_v, preds) ** 0.5

        sampler = optuna.samplers.TPESampler(seed=random_state)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        print(f"Running Optuna TPE with {n_trials} trials on {len(X_t)} samples")
        study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        best.update(fixed)
        print(f"Best parameters: {best}")
        print(f"Best RMSE: {study.best_value:.3f}")

        self.best_model = xgb.XGBRegressor(
            objective=objective_val,
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method=tree_method,
            **best
        )
        pkl.dump(study, open(self.path, "wb"))
        return best

    def train(self, input_columns, X, y, parameters={}):
        """Trains the model using the best hyperparameters found."""
        if self.best_model:
            if DEVICE =='cuda':
                self.best_model.fit(X, y,tree_method='gpu_hist')
            else:
                self.best_model.fit(X, y)
            # print(f"The optimal number of trees is {self.best_model.best_iteration}")
            # feature importance
            imp, feats = zip(*sorted(zip(self.best_model.feature_importances_, input_columns)))

            # plot
            pyplot.barh(feats, imp)
            pyplot.show()
        else:
            eta = parameters['eta'][0]
            max_depth =parameters['max_depth'][0]
            n_estimators = parameters['n_estimators'][0]
            print(f"Using user-defined hyperparameters, eta: {eta}, max_depth: {max_depth}, n_estimators: {n_estimators}")
            self.best_model = self.model
            if DEVICE =='cuda':
                self.best_model.fit(X, y,tree_method='gpu_hist')
            else:
                self.best_model.fit(X, y)
            # print(f"The optimal number of trees is {self.best_model.best_iteration}")
            # feature importance
            imp, feats = zip(*sorted(zip(self.best_model.feature_importances_, input_columns)))

            # plot
            pyplot.barh(feats, imp)
            pyplot.show()

    def predict(self, X):
        """Predicts using the trained XGBoost model on the provided data."""
        if self.best_model:
            return self.best_model.predict(X)
        else:
            print("Model is not trained yet. Please train the model first.")
            return None

    def evaluate(self, X, y):
        """Evaluates the trained model on a separate test set."""
        if self.best_model:
            
            # define model evaluation method
            cv = RepeatedKFold(n_splits=10, n_repeats=3)
            
            # evaluate model
            scores = cross_val_score(self.best_model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

            # Caculate model performance and force scores to be positive
            print('Mean MAE: %.3f (%.3f)' % (abs(scores.mean()), scores.std()) )

        else:
            print("Model is not trained yet. Please train the model first.")


def XGB_Train(model_path, input_columns, x_train, y_train, tries, hyperparameters, perc_data, optimize='bayesian', n_trials=50):
    start_time = time.time()

    # Start running the model several times.
    for try_number in range(tries):
        print(f'Trial Number {try_number} ==========================================================')

        # # Set the optimizer, create the model, and train it.
        xgboost_model = XGBoostRegressorCV(hyperparameters, f"{model_path}/best_model_hyperparameters.pkl")

        if optimize in ('gridsearch', 'bayesian'):
            new_data_len = int(len(x_train) * perc_data)
            print(f"Tuning hyperparameters on {perc_data*100}% of training data")
            x_hyper, y_hyper = x_train.iloc[:new_data_len], y_train.iloc[:new_data_len]

            if optimize == 'bayesian':
                best_params = xgboost_model.tune_hyperparameters(x_hyper, y_hyper, n_trials=n_trials)
            else:
                best_params = xgboost_model.tune_gridsearch(x_hyper, y_hyper)

            print('Training model with optimized hyperparameters')
            xgboost_model.train(input_columns, x_train, y_train)
            print('Saving Model')

        else:
            print('Training model with user-identified hyperparameters')
            xgboost_model.train(input_columns, x_train, y_train, hyperparameters)
            print('Saving Model')
            best_params = hyperparameters
            
        #adjust this to match changing models
        pkl.dump(xgboost_model, open(f"{model_path}/best_model.pkl", "wb"))  

    print('Run is Done!' + "Run Time:" + " %s seconds " % (time.time() - start_time))
    return best_params


def XGB_Predict(model_path, modelname, x_test, y_test, Use_fSCA_Threshold):

    PredDF = x_test.copy()

    #Load model
    xgboost_model = pkl.load(open(f"{model_path}/best_model.pkl", "rb"))
    predictions = xgboost_model.predict(x_test)
    predictions[predictions<0] = 0

    print('Model Predictions complete')

    #connect predictions with feature input dataframe
    predname = f"{modelname}_swe_cm"
    PredDF['ASO_swe_cm'] = y_test
    PredDF[predname] = predictions


    #change lines in predictions to reflect VIIRS hasSnow
    if Use_fSCA_Threshold == True:
        PredDF[predname][PredDF['hasSnow'] == False] = 0

    # #save predictions as compressed pkl file
    # pred_path = f"{HOME}/NWM_ML/Predictions/Hindcast/{modelname}/Multilocation"
    # file_path = f"{pred_path}/{modelname}_predictions.pkl"
    # if os.path.exists(pred_path) == False:
    #     os.makedirs(pred_path)
    # with open(file_path, 'wb') as handle:
    #     pkl.dump(Preds_Dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    return PredDF

def XGB_Perf_Save(df, path, name):
    table = pa.Table.from_pandas(df)
    # Parquet with Brotli compression
    pq.write_table(table, f"{path}/{name}.parquet", compression='BROTLI')
