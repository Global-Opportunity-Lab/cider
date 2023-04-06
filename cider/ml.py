# Copyright ©2022-2023. The Regents of the University of California
# (Regents). All Rights Reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met: 

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer. 

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the 
# documentation and/or other materials provided with the
# distribution. 

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import io
import json
from math import ceil
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd
from helpers.ml_utils import (DropMissing, Winsorizer, auc_overall, load_model,
                              metrics)
from helpers.plot_utils import clean_plot
from helpers.utils import make_dir
from joblib import dump, load  # type: ignore[import]
from lightgbm import LGBMRegressor  # type: ignore[import]
from pandas import DataFrame as PandasDataFrame
from scipy.stats import spearmanr  # type: ignore[import]
from sklearn.compose import ColumnTransformer  # type: ignore[import]
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import]
from sklearn.feature_selection import VarianceThreshold  # type: ignore[import]
from sklearn.impute import SimpleImputer  # type: ignore[import]
from sklearn.linear_model import Lasso  # type: ignore[import]
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV  # type: ignore[import]
from sklearn.model_selection import (KFold, cross_val_predict, cross_val_score,
                                     cross_validate)
from sklearn.pipeline import Pipeline  # type: ignore[import]
from sklearn.preprocessing import MinMaxScaler  # type: ignore[import]
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from skmisc.loess import loess  # type: ignore[import]

from .datastore import DataStore, DataType


class Learner:

    def __init__(self, datastore: DataStore, clean_folders: bool = False, kfold_tune: int = 3, kfold_predict: int = 5) -> None:
        self.cfg = datastore.cfg
        self.ds = datastore
        self.outputs = self.cfg.path.working.directory_path / 'ml'

        # Prepare working directories
        make_dir(self.outputs, clean_folders)
        make_dir(self.outputs / 'outputs')
        make_dir(self.outputs / 'tables')

        self.kfold_tune = KFold(n_splits=kfold_tune, shuffle=True, random_state=100)
        self.kfold_predict = KFold(n_splits=kfold_predict, shuffle=True, random_state=101)


        # Define models
        self.untuned_models = {
            'linear': Pipeline([('dropmissing', DropMissing(threshold=0.9)),
                                ('droplowvariance', VarianceThreshold(threshold=0.01)),
                                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                                ('winsorizer', Winsorizer(limits=(.005, .995))),
                                ('scaler', StandardScaler()),
                                ('model', LinearRegression())]),

            'lasso': Pipeline([('dropmissing', DropMissing(threshold=0.9)),
                               ('droplowvariance', VarianceThreshold(threshold=0.01)),
                               ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                               ('winsorizer', Winsorizer(limits=(.005, .995))),
                               ('scaler', StandardScaler()),
                               ('model', Lasso(alpha=.05))]),

            'ridge': Pipeline([('dropmissing', DropMissing(threshold=0.9)),
                               ('droplowvariance', VarianceThreshold(threshold=0.01)),
                               ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                               ('winsorizer', Winsorizer(limits=(.005, .995))),
                               ('scaler', StandardScaler()),
                               ('model', Ridge(alpha=.05))]),

            'randomforest': Pipeline([('dropmissing', DropMissing(threshold=0.9)),
                                      ('droplowvariance', VarianceThreshold(threshold=0.01)),
                                      ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                                      ('winsorizer', Winsorizer(limits=(.005, .995))),
                                      ('model', RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=1,
                                                                      max_depth=4))]),

            'gradientboosting': Pipeline([('dropmissing', DropMissing(threshold=0.9)),
                                          ('droplowvariance', VarianceThreshold(threshold=0.01)),
                                          ('winsorizer', Winsorizer(limits=(.005, .995))),
                                          ('model', LGBMRegressor(n_estimators=100, n_jobs=-1, random_state=1,
                                                                  min_data_in_leaf=100, num_leaves=4,
                                                                  learning_rate=0.1, verbose=-10))])
        }

        self.tuned_models = {
            'linear': Pipeline([('dropmissing', DropMissing()),
                                ('droplowvariance', VarianceThreshold()),
                                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                                ('winsorizer', Winsorizer()),
                                ('scaler', StandardScaler()),
                                ('model', LinearRegression())]),

            'lasso': Pipeline([('dropmissing', DropMissing()),
                               ('droplowvariance', VarianceThreshold()),
                               ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                               ('winsorizer', Winsorizer()),
                               ('scaler', StandardScaler()),
                               ('model', Lasso())]),

            'ridge': Pipeline([('dropmissing', DropMissing()),
                               ('droplowvariance', VarianceThreshold()),
                               ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                               ('winsorizer', Winsorizer()),
                               ('scaler', StandardScaler()),
                               ('model', Ridge())]),

            'randomforest': Pipeline([('dropmissing', DropMissing()),
                                      ('droplowvariance', VarianceThreshold()),
                                      ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                                      ('winsorizer', Winsorizer()),
                                      ('model', RandomForestRegressor(random_state=1, n_jobs=-1))]),

            'gradientboosting': Pipeline([('dropmissing', DropMissing()),
                                          ('droplowvariance', VarianceThreshold()),
                                          ('winsorizer', Winsorizer()),
                                          ('model', LGBMRegressor(random_state=1, n_jobs=-1, verbose=-10))])
        }

        self.grids = self.ds.cfg.hyperparams

        # Load data into datastore
        data_type_map = {DataType.FEATURES: None,
                         DataType.LABELS: None}
        self.ds.load_data(data_type_map=data_type_map)
        self.ds.merge()

        # Raise warnings if data is sparse
        rows, cols = self.ds.merged.shape
        if rows < 100:
            print("WARNING: The training data has fewer than 100 examples, which will likely result in unreliable "
                  "results and a model with poor predictive performance.")
        if cols < 12:
            print("WARNING: The training data has fewer than 10 features, which could result in a model with poor "
                  "predictive performance")
        sparse_feats = ((pd.isna(self.ds.merged).sum()/rows) > 0.9).sum()/(cols-3)*100
        if sparse_feats > 5:
            print(f"WARNING: {sparse_feats:.2f}% of features have data for less than 10% of users.")

    def untuned_model(self, model_name: str) -> Dict[str, str]:
        """
        Trains the ML model specified by 'model_name' and returns the R2 and the RMSE it obtained on the train and test
        sets. It also saves the trained model and computes its feature importance.

        Args:
            model_name: The name of the model to use - one of ['linear', 'lasso', 'ridge', 'randomforest',
        'gradientboosting']

        Returns: A dict of R2 and RMSE scores obtained on training and test sets.
        """

        make_dir(self.outputs / 'untuned_models' / model_name)

        raw_scores = cross_validate(self.untuned_models[model_name], self.ds.x, self.ds.y,
                                    cv=self.kfold_predict,
                                    return_train_score=True,
                                    scoring=['r2', 'neg_root_mean_squared_error'],
                                    fit_params={'model__sample_weight': self.ds.weights})

        scores = {'train_r2': '%.2f (%.2f)' % (raw_scores['train_r2'].mean(), raw_scores['train_r2'].std()),
                  'test_r2': '%.2f (%.2f)' % (raw_scores['test_r2'].mean(), raw_scores['test_r2'].std()),
                  'train_rmse': '%.2f (%.2f)' % (-raw_scores['train_neg_root_mean_squared_error'].mean(),
                                                 raw_scores['train_neg_root_mean_squared_error'].std()),
                  'test_rmse': '%.2f (%.2f)' % (-raw_scores['test_neg_root_mean_squared_error'].mean(),
                                                raw_scores['test_neg_root_mean_squared_error'].std())}
        with open(self.outputs / 'untuned_models' / model_name / 'results.json', 'w') as f:
            json.dump(scores, f)

        if raw_scores['test_r2'].mean() < 0.1:
            print("WARNING: The R2 score is below 0.1, which is a strong sign of poor predictive performance; it is "
                  "recommended to investigate any data issues before proceeding further.")

        # Save model
        model = self.untuned_models[model_name].fit(self.ds.x, self.ds.y, model__sample_weight=self.ds.weights)
        dump(model, self.outputs / 'untuned_models' / model_name / 'model')

        # Feature importances
        self.feature_importances(model_name=model_name, kind='untuned')

        return scores

    def tuned_model(self, model_name: str) -> Dict[str, str]:
        """
        Trains the ML model specified by 'model_name' and returns the R2 and the RMSE it obtained on the train and test
        sets by all models. During training, it will try the grid of hyper-parameters specified in the config file for
        the corresponding model. It also saves the trained models and computes its their feature importance.

        Args:
            model_name: The name of the model to use - one of ['linear', 'lasso', 'ridge', 'randomforest',
        'gradientboosting']

        Returns: A dict of R2 and RMSE scores obtained on training and test sets by the best model.
        """

        make_dir(self.outputs / 'tuned_models' / model_name)

        model = GridSearchCV(estimator=self.tuned_models[model_name],
                             param_grid=self.grids[model_name],
                             cv=self.kfold_tune,
                             verbose=0,
                             scoring=['r2', 'neg_root_mean_squared_error'],
                             return_train_score=True,
                             refit='r2',
                             n_jobs=-1)

        model.fit(self.ds.x, self.ds.y, model__sample_weight=self.ds.weights)

        # Save tuning results
        tuning_results = pd.DataFrame(model.cv_results_)
        tuning_results.drop(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_fit_time'], axis=1) \
            .to_csv(self.outputs / 'tuned_models' / model_name / 'tuning.csv', index=False)

        # Save accuracy results for best model
        best_model = tuning_results.iloc[tuning_results['mean_test_r2'].argmax()]
        scores = {'train_r2': '%.2f (%.2f)' % (best_model['mean_train_r2'], best_model['std_train_r2']),
                  'test_r2': '%.2f (%.2f)' % (best_model['mean_test_r2'], best_model['std_test_r2']),
                  'train_rmse': '%.2f (%.2f)' % (
                      -best_model['mean_train_neg_root_mean_squared_error'],
                      best_model['std_train_neg_root_mean_squared_error']), 'test_rmse': '%.2f (%.2f)' % (
                -best_model['mean_test_neg_root_mean_squared_error'],
                best_model['std_test_neg_root_mean_squared_error'])}
        with open(self.outputs / 'tuned_models' / model_name / 'results.json', 'w') as f:
            json.dump(scores, f)

        if best_model['mean_test_r2'] < 0.1:
            print(
                "WARNING: The R2 score of the best model is below 0.1, which is a strong sign of poor predictive "
                "performance; it is recommended to investigate any data issues before proceeding further.")

        # Save model
        dump(model, self.outputs / 'tuned_models' / model_name / 'model')

        # Feature importances
        self.feature_importances(model_name=model_name, kind='tuned')

        return scores

    def automl(self, model_name: str) -> None:
        """
        Trains ML models using AutoML - the libraries' parameters are specified in the config file. All models,
        including the ensembles, are saved to disk.

        Args:
            model_name: The name of the AutoML library to use - currently it only supports 'autogluon' (AutoGluon).
        """

        # Make sure model_name is correct, get relevant cfg
        assert model_name in ['autogluon']
        try:
            from autogluon.tabular import TabularPredictor  # type: ignore[import]
        except ModuleNotFoundError:
            raise ImportError(
                "Optional dependency autogluon is required for automl. Please install it (e.g. using pip). "
                "Note that autogluon does not support python 3.9, so you must be using python 3.8 for this "
                "to work."
            )
        make_dir(self.outputs/ 'automl_models' / model_name)

        if model_name == 'autogluon':
            cfg = self.cfg.params.automl.autogluon
            train_data = pd.concat([self.ds.x, self.ds.y, self.ds.weights], axis=1)
            model = TabularPredictor(label=cfg.label,
                                     eval_metric=cfg.eval_metric,
                                     sample_weight=cfg.sample_weight,
                                     weight_evaluation=True,
                                     path=self.outputs / 'automl_models' / model_name / 'model')
            model.fit(train_data,
                      presets='best_quality',
                      auto_stack=True,
                      time_limit=cfg.time_limit,
                      excluded_model_types=['FASTAI'])

            if model.eval_metric == 'r2' and model.leaderboard().loc[0, 'score_val'] < 0.1:
                print(
                    "WARNING: The R2 score of the best model is below 0.1, which is a strong sign of poor predictive "
                    "performance; it is recommended to investigate any data issues before proceeding further.")

        print('Finished automl training!')

    def feature_importances(self, model_name: str, kind: str = 'tuned') -> PandasDataFrame:
        """
        Computes the relative of features for a trained model.

        Args:
            model_name: The name of the model.
            kind: The type of model, i.e. untuned, tuned, or automl.

        Returns: The pandas df with all features and their importance.
        """
        # Load model
        subdir = kind + '_models'
        model_name, model = load_model(model_name, out_path=self.outputs, kind=kind)

        # TODO: improve performance of autogluon's feature importance computation
        if model_name == 'autogluon':
            imports = model.feature_importance(data=self.ds.merged)
            imports = imports.reset_index.rename(columns={'index': 'Feature', 'importance': 'Importance'})
            imports = imports[['Feature', 'Importance']]
        else:
            if 'feature_importances_' in dir(model.named_steps['model']):
                imports = model.named_steps['model'].feature_importances_
            else:
                imports = model.named_steps['model'].coef_
            imports = pd.DataFrame([self.ds.x.columns, imports]).T
            imports.columns = ['Feature', 'Importance']
            
        imports = imports.sort_values('Importance', ascending=False)
        imports.to_csv(self.outputs / subdir / model_name / 'feature_importances.csv', index=False)
        return imports

    def oos_predictions(self, model_name: str, kind: str = 'tuned') -> PandasDataFrame:
        """
        Computes out-of-sample predictions for all training + test samples.

        Args:
            model_name: The name of the model.
            kind: The type of model, i.e. untuned, tuned, or automl.

        Returns: The pandas df with true and predicted values for all training + test samples.

        """
        # Load model
        subdir = kind + '_models'
        model_name, model = load_model(model_name, out_path=self.outputs, kind=kind)
        
        if model_name == 'autogluon':
            oos = model.get_oof_pred()
        else:
            oos = cross_val_predict(model, self.ds.x, self.ds.y, cv=self.kfold_predict)

        oos = pd.DataFrame([list(self.ds.merged['name']), list(self.ds.y), oos]).T
        oos.columns = ['name', 'true', 'predicted']
        oos['weight'] = self.ds.weights
        oos.to_csv(self.outputs / subdir/ model_name / 'oos_predictions.csv', index=False)
        return oos

    def population_predictions(self, model_name: str, kind: str = 'tuned', chunksize: int = 100) -> PandasDataFrame:
        """
        Computes predictions for all population samples, i.e. those samples that do not have ground-truth data.

        Args:
            model_name: The name of the model.
            kind: The type of model, i.e. untuned, tuned, or automl.
            n_chunks: The number of chunks to divide the full population dataset in.

        Returns: The pandas df with predicted values.
        """
        # Load model
        subdir = kind + '_models'
        model_name, model = load_model(model_name, out_path=self.outputs, kind=kind)
        
        features_path = self.ds.features_path
        columns = pd.read_csv(features_path, nrows=1).columns
        
        total_len = len(pd.read_csv(features_path, usecols=['name']))
        n_chunks = ceil(total_len / chunksize)

        results = []
        for chunk in range(n_chunks):
            x = pd.read_csv(features_path, skiprows=1 + chunk * chunksize, nrows=chunksize, header=None)
            x.columns = columns
            results_chunk = x[['name']].copy()
            results_chunk['predicted'] = model.predict(x[self.ds.x.columns])
            results.append(results_chunk)
        results_df = pd.concat(results)

        results_df.to_csv(self.outputs / subdir / model_name / 'population_predictions.csv', index=False)
        return results_df

    def scatter_plot(self, model_name: str, kind: str = 'tuned') -> None:
        """
        Charts the out-of-sample predictions and the true values as a scatter plot, computes the correlation coefficient
        between the two series, and superimposed the fitted LOESS curve.

        Args:
            model_name: The name of the model.
            kind: The type of model, i.e. untuned, tuned, or automl.
        """
        # Load model
        subdir = kind + '_models'
        oos = pd.read_csv(self.outputs / subdir / model_name / 'oos_predictions.csv')
        oos['weight'] = 100 * ((oos['weight'] - oos['weight'].min()) / (oos['weight'].max() - oos['weight'].min()))
        oos_repeat = pd.DataFrame(np.repeat(oos.values, oos['weight'], axis=0), columns=oos.columns).astype(oos.dtypes)
        corr = np.corrcoef(oos_repeat['true'], oos_repeat['predicted'])[0][1]

        grid = np.linspace(oos['true'].min(), oos['true'].max(), 100)
        l = loess(list(oos['true']), list(oos['predicted']))
        l.fit()
        pred = l.predict(grid, stderror=True)
        conf = pred.confidence()
        lowess = pred.values

        fig, ax = plt.subplots(1, figsize=(10, 7))

        ax.scatter(oos['true'], oos['predicted'], s=oos['weight'], label='Data', color='indianred')
        ax.plot(grid, lowess, color='mediumseagreen', label='LOESS Fit', linewidth=3)
        ax.fill_between(grid, conf.lower, conf.upper, color='mediumseagreen', alpha=0.1)

        ax.set_xlim(grid.min(), grid.max())
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        ax.set_title(f'True vs. Predicted Values (r = {corr})')
        ax.legend(loc='best')
        clean_plot(ax)

        plt.savefig(self.outputs / subdir / model_name / 'scatterplot.png', dpi=300)
        plt.show()

    def feature_importances_plot(
        self,
        model_name: str, 
        kind: str = 'tuned', 
        n_features: int = 20,
        plot_title: str = 'Feature Importances'
    ) -> None:
        """
        Produces horizontal bar plots of already calculated feature importances.

        Args:
            model_name: The name of the model.
            kind: The type of model, i.e. untuned, tuned, or automl.
            n_features: The number of top features to use in the plot.
        """
        # Load model
        subdir =  kind + '_models'
        importances = pd.read_csv(self.outputs / subdir / model_name / 'feature_importances.csv')

        importances = importances.sort_values('Importance', ascending=False)
        importances = importances[:n_features].sort_values('Importance', ascending=True)

        importances['color'] = importances['Feature'] \
            .apply(lambda x: 'indianred' if x.split('_')[0] in ['cdr', 'international']
                   else 'mediumseagreen' if x.split('_')[0] == 'location'
                   else 'darkorange' if x.split('_')[0] == 'mobiledata'
                   else 'dodgerblue' if x.split('_')[0] == 'mobilemoney'
                   else 'orchid' if x.split('_')[0] == 'recharges'
                   else 'grey')

        importances['Feature'] = importances['Feature'] \
            .apply(lambda x: ' '.join(x.split('_'))
                   .replace('percent', '%')
                   .replace('callandtext', '')
                   .replace('weekday', 'WD')
                   .replace('weekend', 'WE')
                   .replace('allday', '')
                   .replace('allweek', ''))

        fig, ax = plt.subplots(1, figsize=(20, 10))

        ax.barh(importances['Feature'], importances['Importance'], color=importances['color'])

        ax.set_title(plot_title, fontsize='large')
        ax.set_xlabel('Feature Importance')
        clean_plot(ax)

        plt.savefig(self.outputs / subdir / model_name / 'feature_importances.png', dpi=300)
        plt.show()

    def targeting_table(self, model_name: str, kind: str = 'tuned') -> PandasDataFrame:
        """
        Computes classification metrics using OOS predictions and when targeting 10 different subsets of the population,
        the bottom p percentile where p = [0, 10, ..., 90].

        Args:
            model_name: The name of the model.
            kind: The type of model, i.e. untuned, tuned, or automl.

        Returns: The pandas df with correlation, accuracy, precision, recall, AUC, when targeting the bottom p
        percentile of the population, where p = [0, 10, ..., 90].
        """

        subdir = kind + '_models'
        try:
            oos = pd.read_csv(self.outputs / subdir / model_name / 'oos_predictions.csv')
            
        except FileNotFoundError:
            raise ValueError('Must perform oos prediction in order to compile targeting table') # TODO(leo): clarify this error

        oos['weight'] = 100 * (oos['weight'] - oos['weight'].min()) / (oos['weight'].max() - oos['weight'].min())
        oos_repeat = pd.DataFrame(np.repeat(oos.values, oos['weight'], axis=0), columns=oos.columns).astype(oos.dtypes)

        grid = np.linspace(0, 90, 10)[1:]
        metric_grid = [metrics(oos_repeat['true'], oos_repeat['predicted'], p) for p in grid]

        table = pd.DataFrame()
        table['Proportion of Population Targeted'] = [('%i' % p) + '%' for p in grid]
        table['Pearson'] = np.corrcoef(oos_repeat['true'], oos_repeat['predicted'])[0][1]
        table['Spearman'] = spearmanr(oos_repeat['true'], oos_repeat['predicted'])[0]
        table['AUC'] = auc_overall(oos_repeat['true'], oos_repeat['predicted'])
        table['Accuracy'] = [('%i' % (g[0] * 100)) + '%' for g in metric_grid]
        table['Precision'] = [('%i' % (g[1] * 100)) + '%' for g in metric_grid]
        table['Recall'] = [('%i' % (g[2] * 100)) + '%' for g in metric_grid]

        table = table.round(2)
        table.to_csv(self.outputs / subdir / model_name / 'targeting_table.csv', index=False)
        return table
