from box import Box
import yaml
from helpers.utils import *
from helpers.plot_utils import *
from helpers.io_utils import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from wpca import WPCA

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

        # Initialize values
        if dataframe is not None:
            self.survey_data = dataframe
        else:
            self.survey_data = pd.read_csv(data + file_names.survey)
        if 'weight' not in self.survey_data.columns:
            self.survey_data['weight'] = 1
        

        # Prepare working directory
        make_dir(outputs, clean_folders)
    
    def asset_index(self, cols, use_weights=True):

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

    def train_pmt(self, cols, cv=5, use_weights=True):

        # TODO
        return False

    def pretrained_pmt(self, cols, use_weights=True):

        # TODO
        return False


    def select_features(self, cols, method='forward_selection', use_weights=True):

        # TODO
        return False





   