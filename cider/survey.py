# TODO: parallelize lasso and forward selection
from datastore import DataStore, DataType
from helpers.utils import check_columns_exist, check_column_types, make_dir, weighted_corr
from helpers.plot_utils import clean_plot
from helpers.ml_utils import Winsorizer
from joblib import dump, load  # type: ignore[import]
from lightgbm import LGBMRegressor  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd
from pandas import DataFrame as PandasDataFrame
from sklearn.compose import ColumnTransformer  # type: ignore[import]
from sklearn.decomposition import PCA  # type: ignore[import]
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import]
from sklearn.linear_model import Lasso, LinearRegression, Ridge  # type: ignore[import]
from sklearn.metrics import r2_score  # type: ignore[import]
from sklearn.model_selection import cross_validate, cross_val_predict, GridSearchCV  # type: ignore[import]
from sklearn.pipeline import Pipeline  # type: ignore[import]
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler  # type: ignore[import]
from typing import List, Optional, Tuple, Union
from wpca import WPCA  # type: ignore[import]


class SurveyOutcomeGenerator:

    def __init__(self,
                 datastore: DataStore,
                 dataframe: Optional[PandasDataFrame] = None,
                 clean_folders: bool = False) -> None:
        self.cfg = datastore.cfg
        self.ds = datastore
        self.outputs = datastore.outputs + 'survey/'
        self.continuous = self.cfg.col_types.survey.continuous
        self.categorical = self.cfg.col_types.survey.categorical
        self.binary = self.cfg.col_types.survey.binary

        # Prepare working directories
        make_dir(self.outputs, clean_folders)

        # Get hyper-parameter grids
        self.grids = self.ds.cfg.hyperparams
        for key1 in self.grids.keys():
            grid = {}
            for key2 in self.grids[key1].keys():
                if 'variance' not in key2 and 'missing' not in key2 and 'winsorizer' not in key2:
                    grid[key2] = self.grids[key1][key2]
            self.grids[key1] = grid

        # Load data into datastore
        data_type_map = {DataType.SURVEY_DATA: dataframe}
        self.ds.load_data(data_type_map=data_type_map)

    def asset_index(self, cols: List[str], use_weights: bool = True) -> PandasDataFrame:
        """
        Derives an asset index - first component of (weighted) PCA - from specified set of columns.

        Args:
            cols: Columns (i.e. survey questions) to use to derive the index.
            use_weights: If true, uses a weighted PCA.

        Returns: Pandas df containing asset index of surveyed users.
        """
        # Prepare working directory
        out_subdir = self.outputs + '/asset_index'
        make_dir(out_subdir, True)

        # Check that categorical/binary columns are not being included
        if len(set(cols).intersection(set(self.categorical))) > 0:
            print('Warning: %i columns are categorical but will be treated as continuous for the purpose of the '
                  'asset index.' % len(set(cols).intersection(set(self.categorical))))
        if len(set(cols).intersection(set(self.binary))):
            print(
                'Warning: %i columns are binary but will be treated as continuous for the purpose of the asset index.' %
                len(set(cols).intersection(set(self.binary))))

            # Drop observations with null values
        n_obs = len(self.ds.survey_data)
        assets = self.ds.survey_data.dropna(subset=cols)
        dropped = n_obs - len(assets)
        if dropped > 0:
            print('Warning: Dropping %i observations with missing values (%i percent of all observations)' % (
                dropped, 100 * dropped / n_obs))

        # Scale data
        scaler = MinMaxScaler()
        scaled_assets = scaler.fit_transform(assets[cols])

        # Calculate asset index and basis vector
        if use_weights:
            np.random.seed(2)
            pca = WPCA(n_components=1)
            w = np.vstack([assets['weight'].values for i in range(len(cols))]).T
            asset_index = pca.fit_transform(scaled_assets, weights=w)[:, 0]
        else:
            pca = PCA(n_components=1, random_state=1)
            asset_index = pca.fit_transform(scaled_assets)[:, 0]
        print('PCA variance explained: %.2f' % (100 * pca.explained_variance_ratio_[0]) + '%')

        # Write asset index to file
        asset_index = pd.DataFrame([list(assets['unique_id']), asset_index]).T
        asset_index.columns = ['unique_id', 'asset_index']
        self.index = asset_index
        asset_index.to_csv(out_subdir + '/index.csv', index=False)

        # Write basis vector to file
        basis_vector = pd.DataFrame([cols, pca.components_[0]]).T
        basis_vector.columns = ['Item', 'Magnitude']
        basis_vector = basis_vector.sort_values('Magnitude', ascending=False)
        self.basis_vector = basis_vector
        basis_vector.to_csv(out_subdir + '/basis_vector.csv', index=False)

        return asset_index

    def fit_pmt(self, outcome: str, cols: List[str], model_name: str = 'linear', kfold: int = 5,
                use_weights: bool = True, log_transform: bool = False, scale: bool = False, winsorize: bool = False
                ) -> PandasDataFrame:
        """
        Trains a model to predict an outcome variable using a set of survey questions - this will be the PMT.
        Also provides metrics on modelling performance.

        Args:
            outcome: Target variable.
            cols: Independent variables.
            model_name: ML model to use - can be one of ['linear', 'lasso', 'ridge', 'randomforest', 'gradientboosting']
            kfold: Number of cross-validation folds to use.
            use_weights: If true, uses survey weights.
            log_transform: If true, log transforms outcome variable.
            scale: If true, standard scales inputs.
            winsorize: If ture, winsorizes (.005, .995) inputs.

        Returns: Pandas df of outcome variable, in-sample predictions, and out-of-sample predictions.
        """
        # Prepare working directory
        out_subdir = self.outputs + '/pmt_' + model_name
        make_dir(out_subdir, True)

        # Check that columns are typed correctly
        check_column_types(self.ds.survey_data[cols], continuous=self.continuous, categorical=self.categorical,
                           binary=self.binary)

        # Drop observations with null values
        data = self.ds.survey_data[['unique_id', 'weight', outcome] + cols]
        n_obs = len(data)
        data = data.dropna(subset=[outcome] + list(set(cols).intersection(set(self.continuous + self.binary))))
        dropped = n_obs - len(data)
        if dropped > 0:
            print(
                'Warning: Dropping %i observations with missing values in continuous or binary columns or the outcome '
                '(%i percent of all observations)' %
                (dropped, 100 * dropped / n_obs))

        # Log transform outcome variable if requested
        if log_transform:
            data[outcome] = np.log(data[outcome])

        # Define preprocessing pipelines
        if scale and winsorize:
            continuous_transformer = Pipeline([('winsorizer', Winsorizer(limits=(.005, .995))),
                                               ('scaler', StandardScaler())])
        elif winsorize:
            continuous_transformer = Pipeline([('winsorizer', Winsorizer(limits=(.005, .995)))])
        elif scale:
            continuous_transformer = Pipeline([('scaler', StandardScaler())])
        else:
            continuous_transformer = Pipeline([('null', 'passthrough')])
        categorical_transformer = OneHotEncoder(drop=None, handle_unknown='ignore')
        preprocessor = ColumnTransformer(
            [('continuous', continuous_transformer, list(set(self.continuous).intersection(set(cols)))),
             ('categorical', categorical_transformer, list(set(self.categorical).intersection(set(cols)))),
             ('binary', 'passthrough', list(set(self.binary).intersection(set(cols))))],
            sparse_threshold=0)

        # Compile model
        models = {
            'linear': LinearRegression(),
            'lasso': Lasso(),
            'ridge': Ridge(),
            'randomforest': RandomForestRegressor(random_state=1, n_jobs=-1),
            'gradientboosting': LGBMRegressor(random_state=1, n_jobs=-1, verbose=-10)
        }
        model = Pipeline([('preprocessor', preprocessor), ('model', models[model_name])])
        if model != 'linear':
            model = GridSearchCV(estimator=model,
                                 param_grid=self.grids[model_name],
                                 cv=kfold,
                                 verbose=0,
                                 scoring='r2',
                                 refit='r2',
                                 n_jobs=-1)

        # Fit and save model
        if use_weights:
            model.fit(data[cols], data[outcome], model__sample_weight=data['weight'])
        else:
            model.fit(data[cols], data[outcome])
        if model != 'linear':
            model = model.best_estimator_
        dump(model, out_subdir + '/model')

        # Save feature importances
        if 'feature_importances_' in dir(model.named_steps['model']):
            imports = model.named_steps['model'].feature_importances_
        else:
            imports = model.named_steps['model'].coef_

        colnames = list(pd.get_dummies(data[cols], columns=self.categorical, dummy_na=True, drop_first=False,
                                       prefix_sep='=').columns)
        imports = pd.DataFrame([colnames, imports]).T
        imports.columns = ['Feature', 'Importance']
        imports = imports.sort_values('Importance', ascending=False)
        imports.to_csv(out_subdir + '/feature_importances.csv', index=False)

        # Get in sample and out of sample predictions
        insample = model.predict(data[cols])
        oos = cross_val_predict(model, data[cols], data[outcome], cv=kfold)
        predictions = pd.DataFrame([data['unique_id'].values, insample, oos]).T
        predictions.columns = ['unique_id', 'in_sample_prediction', 'out_of_sample_prediction']
        predictions = predictions.merge(data[['unique_id', 'weight', outcome] + cols], on='unique_id')
        predictions.to_csv(out_subdir + '/predictions.csv', index=False)
        if use_weights:
            r2 = r2_score(predictions[outcome], predictions['in_sample_prediction'],
                          sample_weight=predictions['weight'])
        else:
            r2 = r2_score(predictions[outcome], predictions['in_sample_prediction'])
        print('R2 score: %.2f' % r2)
        return predictions

    def pretrained_pmt(self, other_data_in: Union[str, PandasDataFrame], cols: List[str],
                       model_name: str, dataset_name: str = 'other_data') -> PandasDataFrame:
        """
        Use a pre-trained PMT model to predict the outcome variable in another dataset - check that the latter has the
        same columns and data types of the original survey that was used to train the PMT

        Args:
            other_data_in: new dataset to get predictions for, either as path or as pandas df
            cols: input columns
            model_name: PMT model to use
            dataset_name: name of new dataset

        Returns: pandas df of user id and predicted outcome variable
        """

        # Prepare working directory
        out_subdir = self.outputs + '/pmt_' + model_name + '/' + dataset_name
        make_dir(out_subdir, True)

        # Load data
        if isinstance(other_data_in, str):
            other_data = pd.read_csv(other_data_in)
        else:
            other_data = other_data_in

        # Check that all columns are present and check column types
        original_data = pd.read_csv(self.outputs + '/pmt_' + model_name + '/predictions.csv')
        check_columns_exist(original_data, cols, 'training dataset')
        check_columns_exist(other_data, cols, 'prediction dataset')
        check_column_types(other_data[cols], continuous=self.continuous, categorical=self.categorical,
                           binary=self.binary)

        # Drop observations with null values
        other_data = other_data[['unique_id'] + cols]
        n_obs = len(other_data)
        other_data = other_data.dropna(subset=list(set(cols).intersection(set(self.continuous + self.binary))))
        dropped = n_obs - len(other_data)
        if dropped > 0:
            print('Warning: Dropping %i observations with missing values in continuous or binary columns '
                  '(%i percent of all observations)' % (dropped, 100 * dropped / n_obs))

        # Check that ranges are the same as training data
        for c in set(cols).intersection(set(self.categorical)):
            set_dif = set(other_data[c].dropna()).difference(set(original_data[c].dropna()))
            if len(set_dif) > 0:
                print('Warning: There are values in categorical column ' + c +
                      ' that are not present in training data; they will not be positive for any dummy column. '
                      'Values: ' + ','.join([str(x) for x in set_dif]))
        for c in set(cols).intersection(set(self.continuous)):
            if np.round(other_data[c].min(), 2) < np.round(original_data[c].min(), 2) or \
                    np.round(other_data[c].max(), 2) > np.round(original_data[c].max(), 2):
                print('Warning: There are values in continuous column ' + c +
                      ' that are outside of the range in the training data; the original standardization will apply.')

        # Load and apply model, save predictions
        model = load(self.outputs + '/pmt_' + model_name + '/model')
        predictions = pd.DataFrame([other_data['unique_id'].values, model.predict(other_data[cols])]).T
        predictions.columns = ['unique_id', 'prediction']
        predictions = predictions.merge(other_data, on='unique_id')
        predictions.to_csv(out_subdir + '/predictions.csv', index=False)
        return predictions

    def select_features(self, outcome: str, cols: List[str], n_features: int, method: str = 'correlation',
                        model_name: str = '', kfold: int = 5, use_weights: bool = True, log_transform: bool = False,
                        plot: bool = True) -> Tuple[List[str], PandasDataFrame]:
        """
        Selects the top-N features that are most useful to predict the outcome variable.

        Args:
            outcome: Target variable.
            cols: Independent variables.
            n_features: Number of features to select.
            method: Method to use to select the top-N features - it can be:
                'correlation': return top N features most correlated with the outcome
                'lasso': use LASSO regressions for a grid of penalties to select features, use the one which has the
                 closest to n_features features
                OTHER STRING: forward selection with a machine learning model
            model_name: Name to use when saving outputs.
            kfold: Number of cross-validation folds to use.
            use_weights: If true, uses survey weights.
            log_transform: If true, log transforms outcome variable.
            plot: If true, plots performance metrics.

        Returns: Top-N features selected, pandas df of scores.
        """

        # Prepare working directory
        out_subdir = self.outputs + 'feature_selection'
        make_dir(out_subdir, True)

        # Correlation: Return top N features most correlated with the outcome
        if method == 'correlation':

            correlations = []
            for c in cols:
                subset = self.ds.survey_data[[outcome, c, 'weight']].dropna()
                if use_weights:
                    correlations.append(weighted_corr(subset[outcome].values.flatten(), subset[c].values.flatten(),
                                                      subset['weight'].values.flatten()))
                else:
                    correlations.append(np.corrcoef(subset[outcome], subset[c])[0][1])
            correlations_df = pd.DataFrame([cols, correlations]).T
            correlations_df.columns = ['column', 'correlation']
            correlations_df['abs_value_correlation'] = np.abs(correlations_df['correlation'])
            correlations_df = correlations_df.sort_values('abs_value_correlation', ascending=False)
            correlations_df.to_csv(out_subdir + '/correlations.csv')
            return list(correlations_df[:n_features]['column']), correlations_df.reset_index()

        # LASSO: Use LASSO regressions for a grid of penalties to select features, use the one which has the closest
        # to n_features features
        elif method == 'lasso':

            # Drop observations with null values
            data = self.ds.survey_data[['unique_id', 'weight', outcome] + cols]
            n_obs = len(data)
            data = data.dropna(subset=[outcome] + list(set(cols).intersection(set(self.continuous + self.binary))))
            dropped = n_obs - len(data)
            if dropped > 0:
                print(
                    'Warning: Dropping %i observations with missing values in continuous or binary columns or the '
                    'outcome (%i percent of all observations)' %
                    (dropped, 100 * dropped / n_obs))

            # Log transform outcome variable if requested
            if log_transform:
                data[outcome] = np.log(data[outcome])

            # Define preprocessing pipelines
            continuous_transformer = Pipeline([('scaler', StandardScaler())])
            categorical_transformer = OneHotEncoder(drop=None, handle_unknown='ignore')
            preprocessor = ColumnTransformer(
                [('continuous', continuous_transformer, list(set(self.continuous).intersection(set(cols)))),
                 ('categorical', categorical_transformer, list(set(self.categorical).intersection(set(cols)))),
                 ('binary', 'passthrough', list(set(self.binary).intersection(set(cols))))],
                sparse_threshold=0)

            # Run LASSO regressions
            alphas = np.linspace(0, 1, 100)[1:]
            train_scores, test_scores, features = [], [], []
            if use_weights:
                weights = data['weight'].values
            else:
                weights = np.ones(len(data))
            for alpha in alphas:
                # Get r2 score over cross validation
                lasso = Pipeline([('preprocessor', preprocessor), ('model', Lasso(alpha=alpha))])
                results = cross_validate(lasso, data[cols], data[outcome], return_train_score=True,
                                         fit_params={'model__sample_weight': weights}, cv=kfold)
                train_scores.append(np.mean(results['train_score']))
                test_scores.append(np.mean(results['test_score']))
                # Get nonzero features and importances
                lasso.fit(data[cols], data[outcome], model__sample_weight=weights)
                imports = lasso.named_steps['model'].coef_
                colnames = list(pd.get_dummies(data[cols], columns=self.categorical, dummy_na=True, drop_first=False,
                                               prefix_sep='=').columns)
                imports = pd.DataFrame([colnames, imports]).T
                imports.columns = ['Feature', 'Coefficient']
                imports = imports.sort_values('Coefficient', ascending=False)
                imports = imports[imports['Coefficient'] != 0]
                imports['feature_without_dummies'] = imports['Feature'].apply(lambda x: str(x).split('=')[0])
                features.append(imports)

            num_feats = [len(feats['feature_without_dummies'].unique()) for feats in features]
            r2_df = pd.DataFrame([alphas, num_feats, train_scores, test_scores]).T
            r2_df.columns = ['alpha', 'num_features', 'train_score', 'test_score']
            r2_df.to_csv(out_subdir + '/lasso_r2.csv', index=False)

            # Generate plot
            if plot:
                fig, ax = plt.subplots(1, figsize=(10, 8))
                ax.scatter(num_feats, train_scores, color='mediumseagreen', label='Train')
                ax.scatter(num_feats, test_scores, color='indianred', label='Test')
                ax.set_xlabel('Number of Features')
                ax.set_ylabel('r2 Score')
                ax.set_title('LASSO-based Selection of Features for PMT')
                plt.legend(loc='best')
                clean_plot(ax)
                plt.savefig(out_subdir + '/lasso_r2.png', dpi=300)
                plt.show()

            # Get LASSO regression with closest to correct number of features and write to file
            best_idx = np.argmin(np.abs(np.array(num_feats) - n_features))
            features[best_idx].to_csv(out_subdir + '/lasso_model.csv', index=False)
            return list(set(features[best_idx]['feature_without_dummies'])), test_scores[best_idx], alphas[
                best_idx], r2_df

        # Forward selection with a machine learning model
        else:

            # Drop observations with null values
            data = self.ds.survey_data[['unique_id', 'weight', outcome] + cols]
            n_obs = len(data)
            data = data.dropna(subset=[outcome] + list(set(cols).intersection(set(self.continuous + self.binary))))
            dropped = n_obs - len(data)
            if dropped > 0:
                print('Warning: Dropping %i observations with missing values in continuous or binary columns or the '
                      'outcome (%i percent of all observations)' %
                      (dropped, 100 * dropped / n_obs))

            # Log transform outcome variable if requested
            if log_transform:
                data[outcome] = np.log(data[outcome])

            # Stepwise forward selection
            used_cols: List[str] = []
            unused_cols, train_scores, test_scores = cols, [], []
            if use_weights:
                weights = data['weight']
            else:
                weights = np.ones(len(data))
            for i in range(len(cols)):
                potential_model_test_scores, potential_model_train_scores = [], []
                for c in unused_cols:
                    continuous_transformer = Pipeline([('scaler', StandardScaler())])
                    categorical_transformer = OneHotEncoder(drop=None, handle_unknown='ignore')
                    preprocessor = ColumnTransformer([('continuous', continuous_transformer,
                                                       list(set(self.continuous).intersection(set(used_cols + [c])))),
                                                      ('categorical', categorical_transformer,
                                                       list(set(self.categorical).intersection(set(used_cols + [c])))),
                                                      ('binary', 'passthrough',
                                                       list(set(self.binary).intersection(set(used_cols + [c]))))],
                                                     sparse_threshold=0)
                    model = Pipeline([('preprocessor', preprocessor), ('model', method)])
                    results = cross_validate(model, data[used_cols + [c]], data[outcome], return_train_score=True,
                                             fit_params={'model__sample_weight': weights}, cv=kfold)
                    potential_model_train_scores.append(np.mean(results['train_score']))
                    potential_model_test_scores.append(np.mean(results['test_score']))

                best_idx = np.argmax(potential_model_test_scores)
                best_feature = unused_cols[best_idx]
                used_cols.append(best_feature)
                train_scores.append(potential_model_train_scores[best_idx])
                test_scores.append(potential_model_test_scores[best_idx])
                unused_cols = list(set(unused_cols) - set([best_feature]))

            # Generate plot
            if plot:
                fig, ax = plt.subplots(1, figsize=(10, 8))
                ax.scatter(range(len(used_cols)), train_scores, color='mediumseagreen')
                ax.plot(range(len(used_cols)), train_scores, color='mediumseagreen', label='Train')
                ax.scatter(range(len(used_cols)), test_scores, color='indianred')
                ax.plot(range(len(used_cols)), test_scores, color='indianred', label='Test')
                ax.set_xlabel('Number of Features')
                ax.set_ylabel('r2 Score')
                ax.set_title('Forward Selection of Features for PMT')
                plt.legend(loc='best')
                clean_plot(ax)
                plt.savefig(out_subdir + '/forward_selection' + model_name + '.png', dpi=300)
                plt.show()

            # Return values
            scores = pd.DataFrame([used_cols, train_scores, test_scores]).T
            scores.columns = ['Column added', 'Train score', 'Test Score']
            scores.to_csv(out_subdir + '/forward_selection_' + model_name + '.csv', index=False)
            return used_cols[:n_features], scores
