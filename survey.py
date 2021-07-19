from box import Box
import yaml
from helpers.utils import *
from helpers.plot_utils import *
from helpers.io_utils import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from wpca import WPCA
from helpers.ml_utils import *

class SurveyOutcomeGenerator:

    def __init__(self, cfg_dir, dataframe=None, clean_folders=False):

        # Read config file
        with open(cfg_dir, "r") as ymlfile:
            cfg = Box(yaml.safe_load(ymlfile))
        self.cfg = cfg
        data = cfg.path.survey.data
        outputs = cfg.path.survey.outputs
        self.outputs = outputs
        file_names = cfg.path.survey.file_names
        self.grids = cfg.hyperparams

        # Initialize values
        if dataframe is not None:
            self.survey_data = dataframe
        else:
            self.survey_data = pd.read_csv(data + file_names.survey)
        if 'weight' not in self.survey_data.columns:
            self.survey_data['weight'] = 1

        # Get columns types
        self.continuous = cfg.col_types.survey.continuous
        self.categorical = cfg.col_types.survey.categorical
        self.binary = cfg.col_types.survey.binary
        

        # Prepare working directory
        make_dir(outputs, clean_folders)
    
    def asset_index(self, cols, use_weights=True):

        # Check that categorical/binary columns are not being included
        if len(set(cols).intersection(set(self.categorical))) > 0:
            print('Warning: %i columns are categorical but will be treated as continuous for the purpose of the asset index.' % 
                len(set(cols).intersection(set(self.categorical))))
        if len(set(cols).intersection(set(self.binary))):
            print('Warning: %i columns are binary but will be treated as continuous for the purpose of the asset index.' % 
                len(set(cols).intersection(set(self.binary)))) 

        # Drop observations with null values
        n_obs = len(self.survey_data)
        assets = self.survey_data.dropna(subset=cols)
        dropped = n_obs - len(assets)
        if  dropped > 0:
            print('Warning: Dropping %i observations with missing values (%i percent of all observations)' % (dropped, 100*dropped/n_obs))
        
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
        print('PCA variance explained: %.2f' % (100*pca.explained_variance_ratio_[0]) + '%')

        # Write asset index to file
        asset_index = pd.DataFrame([list(assets['unique_id']), asset_index]).T
        asset_index.columns = ['unique_id', 'asset_index']
        self.index = asset_index
        asset_index.to_csv(self.outputs + '/asset_index.csv', index=False)

        # Write basis vector to file
        basis_vector = pd.DataFrame([cols, pca.components_[0]]).T
        basis_vector.columns = ['Item', 'Magnitude']
        basis_vector = basis_vector.sort_values('Magnitude', ascending=False)
        self.basis_vector = basis_vector
        basis_vector.to_csv(self.outputs + '/basis_vector.csv', index=False)

        return asset_index

    def fit_pmt(self, outcome, cols, model_name='linear', kfold=5, use_weights=True, scale=False, winsorize=False):

        # Drop observations with null values
        data = self.survey_data[['unique_id', 'weight', outcome] + cols]
        n_obs = len(data)
        data = data.dropna(subset=[outcome] + list(set(cols).intersection(set(self.continuous + self.binary))))
        dropped = n_obs - len(data)
        if  dropped > 0:
            print('Warning: Dropping %i observations with missing values in continuous or binary columns or the outcome (%i percent of all observations)' % 
                (dropped, 100*dropped/n_obs))

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
        preprocessor = ColumnTransformer([('continuous', continuous_transformer, list(set(self.continuous).intersection(set(cols)))), 
                                        ('categorical', categorical_transformer, list(set(self.categorical).intersection(set(cols))))])

        # Compile model
        models = {
            'linear': LinearRegression(),
            'lasso': Lasso(),
            'ridge': Ridge(),
            'randomforest': RandomForestRegressor(random_state=1, n_jobs=-1),
            'gradientboosting': LGBMRegressor(random_state=1, n_jobs=-1, verbose=-10)
        }
        model = Pipeline([('preprocessor', preprocessor), ('model', models[model_name])])
        # TODO: Implement functionality for hyperparameter tuning
        #if model != 'linear':
        #    model = GridSearchCV(estimator=model,
        #                     param_grid=self.grids[model_name],
        #                     cv=kfold,
        #                     verbose=0,
        #                     scoring='r2',
        #                     refit='r2',
        #                     n_jobs=-1)


        # Fit model, save feature importances and model
        if use_weights:
            model.fit(data[cols], data[outcome], model__sample_weight=data['weight'])
        else:
             model.fit(data[cols], data[outcome])
        dump(model, self.outputs + '/' + model_name)

        # Get in sample and out of sample predictions
        insample = model.predict(data[cols])
        oos = cross_val_predict(model, data[cols], data[outcome], cv=kfold)
        predictions = pd.DataFrame([data['unique_id'].values, data['weight'].values, data[outcome].values, insample, oos]).T
        predictions.columns = ['unique_id', 'weight', outcome, 'in_sample_prediction', 'out_of_sample_prediction']
        predictions.to_csv(self.outputs + '/' + model_name + '_predictions.csv', index=False)
        print('R2 score: %.2f' % r2_score(predictions[outcome], predictions['in_sample_prediction']))


    def pretrained_pmt(self, cols, use_weights=True):

        # TODO
        return False


    def select_features(self, cols, method='forward_selection', use_weights=True):

        # TODO
        return False





   