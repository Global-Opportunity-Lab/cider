from abc import ABC, abstractmethod
from box import Box
from collections import defaultdict
from geopandas import GeoDataFrame  # type: ignore[import]
from enum import Enum
import inspect
from helpers.io_utils import load_antennas, load_shapefile, load_cdr, load_mobilemoney, load_mobiledata, load_recharges
from helpers.opt_utils import generate_user_consent_list
from helpers.utils import get_project_root, get_spark_session, filter_dates_dataframe, make_dir, save_df
# from helpers.utils import get_project_root
import numpy as np
import os
import pandas as pd
from pandas import DataFrame as PandasDataFrame, Series
from pyspark.sql import DataFrame as SparkDataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import col, count, countDistinct, lit
from typing import Callable, Dict, List, Mapping, Optional, Union
import yaml


class DataType(Enum):
    CDR = 0
    ANTENNAS = 1
    RECHARGES = 2
    MOBILEDATA = 3
    MOBILEMONEY = 4
    SHAPEFILES = 5
    HOME_GROUND_TRUTH = 6
    POVERTY_SCORES = 7
    FEATURES = 8
    LABELS = 9
    TARGETING = 10
    FAIRNESS = 11
    RWI = 12
    SURVEY_DATA = 13


class InitializerInterface(ABC):
    @abstractmethod
    def load_data(self, data_type_map: Dict[DataType, Optional[Union[SparkDataFrame, PandasDataFrame]]]) -> None:
        pass


class DataStore(InitializerInterface):
    def __init__(self, cfg_dir: str):
        # Read config file and store paths
        with open(cfg_dir, "r") as ymlfile:
            cfg = Box(yaml.load(ymlfile, Loader=yaml.FullLoader))
        self.cfg = cfg
        if "root" in cfg.path:
            self.root = cfg.path.root
        else:
            self.root = get_project_root()
        self.data = os.path.join(self.root, self.cfg.path.data)
        outputs = cfg.path.outputs
        self.outputs = outputs
        file_names = cfg.path.file_names
        self.file_names = file_names

        # Create directories
        make_dir(self.outputs + '/datasets/')

        # Parameters
        self.filter_hours = self.cfg.params.home_location.filter_hours
        self.geo = self.cfg.col_names.geo

        # Spark setup
        # TODO(lucio): Initialize spark separately ....
        spark = get_spark_session(cfg)
        self.spark = spark

        # Possible datasets to opt in/out of
        self.datasets = ['cdr', 'cdr_bandicoot', 'recharges', 'mobiledata', 'mobilemoney', 'features']
        # featurizer/home location datasets
        self.cdr: SparkDataFrame
        self.cdr_bandicoot: Optional[SparkDataFrame]
        self.recharges: SparkDataFrame
        self.mobiledata: SparkDataFrame
        self.mobilemoney: SparkDataFrame
        self.antennas: SparkDataFrame
        self.shapefiles: Union[Dict[str, GeoDataFrame]] = {}
        self.home_ground_truth: PandasDataFrame
        self.poverty_scores: PandasDataFrame
        # ml datasets
        self.features: SparkDataFrame
        self.labels: SparkDataFrame
        self.merged: PandasDataFrame
        self.x: PandasDataFrame
        self.y: Series
        self.weights: Series
        # targeting & fairness
        self.targeting: PandasDataFrame
        self.fairness: PandasDataFrame
        # wealth/income maps
        self.rwi: PandasDataFrame
        # survey
        self.survey_data: PandasDataFrame

        # Define mapping between data types and loading methods
        self.data_type_to_fn_map: Dict[DataType, Callable] = {DataType.CDR: self._load_cdr,
                                                              DataType.ANTENNAS: self._load_antennas,
                                                              DataType.RECHARGES: self._load_recharges,
                                                              DataType.MOBILEDATA: self._load_mobiledata,
                                                              DataType.MOBILEMONEY: self._load_mobilemoney,
                                                              DataType.SHAPEFILES: self._load_shapefiles,
                                                              DataType.HOME_GROUND_TRUTH: self._load_home_ground_truth,
                                                              DataType.POVERTY_SCORES: self._load_poverty_scores,
                                                              DataType.FEATURES: self._load_features,
                                                              DataType.LABELS: self._load_labels,
                                                              DataType.TARGETING: self._load_targeting,
                                                              DataType.FAIRNESS: self._load_fairness,
                                                              DataType.RWI: self._load_wealth_map,
                                                              DataType.SURVEY_DATA: self._load_survey}

    def _load_cdr(self, dataframe: Optional[Union[SparkDataFrame, PandasDataFrame]] = None) -> None:
        """
        Load cdr data: use file path specified in config as default, or spark/pandas df
        Args:
            dataframe: spark/pandas df to assign if available
        """
        fpath = self.data + self.file_names.cdr if self.file_names.cdr is not None else None
        if fpath or dataframe is not None:
            print('Loading CDR...')
            cdr = load_cdr(self.cfg, fpath, df=dataframe)
            self.cdr = cdr

    def _load_antennas(self, dataframe: Optional[Union[SparkDataFrame, PandasDataFrame]] = None) -> None:
        """
        Load antennas data: use file path specified in config as default, or spark/pandas df
        Args:
            dataframe: spark/pandas df to assign if available
        """
        fpath = self.data + self.file_names.antennas if self.file_names.antennas is not None else None
        if fpath or dataframe is not None:
            print('Loading antennas...')
            self.antennas = load_antennas(self.cfg, fpath, df=dataframe)

    def _load_recharges(self, dataframe: Optional[Union[SparkDataFrame, PandasDataFrame]] = None) -> None:
        """
        Load recharges data: use file path specified in config as default, or spark/pandas df
        Args:
            dataframe: spark/pandas df to assign if available
        """
        fpath = self.data + self.file_names.recharges if self.file_names.recharges is not None else None
        if fpath or dataframe is not None:
            print('Loading recharges...')
            self.recharges = load_recharges(self.cfg, fpath, df=dataframe)
            print("SUCCESS!")

    def _load_mobiledata(self, dataframe: Optional[Union[SparkDataFrame, PandasDataFrame]] = None) -> None:
        """
        Load mobile data: use file path specified in config as default, or spark/pandas df
        Args:
            dataframe: spark/pandas df to assign if available
        """
        fpath = self.data + self.file_names.mobiledata if self.file_names.mobiledata is not None else None
        if fpath or dataframe is not None:
            print('Loading mobile data...')
            self.mobiledata = load_mobiledata(self.cfg, fpath, df=dataframe)

    def _load_mobilemoney(self, dataframe: Optional[Union[SparkDataFrame, PandasDataFrame]] = None) -> None:
        """
        Load mobile money data: use file path specified in config as default, or spark/pandas df
        Args:
            dataframe: spark/pandas df to assign if available
        """
        fpath = self.data + self.file_names.mobilemoney if self.file_names.mobilemoney is not None else None
        if fpath or dataframe is not None:
            print('Loading mobile data...')
            self.mobilemoney = load_mobilemoney(self.cfg, fpath, df=dataframe)

    def _load_shapefiles(self) -> None:
        """
        Iterate through shapefiles specified in config and load them in self.shapefiles dictionary
        """
        # Load shapefiles
        shapefiles = self.file_names.shapefiles
        for shapefile_fname in shapefiles.keys():
            self.shapefiles[shapefile_fname] = load_shapefile(self.data + shapefiles[shapefile_fname])

    def _load_home_ground_truth(self) -> None:
        """
        Load ground truth data for home locations
        """
        if self.file_names.home_ground_truth is not None:
            self.home_ground_truth = pd.read_csv(self.data + self.file_names.home_ground_truth)
        else:
            print('No ground truth data for home locations has been specified.')

    def _load_poverty_scores(self) -> None:
        """
        Load poverty scores (e.g. those produced by the ML module)
        """
        if self.file_names.poverty_scores is not None:
            self.poverty_scores = pd.read_csv(self.data + self.file_names.poverty_scores)
        else:
            self.poverty_scores = pd.DataFrame()

    def _load_features(self) -> None:
        """
        Load phone usage features to be used for training ML model and subsequent poverty prediction
        """
        self.features = self.spark.read.csv(self.cfg.path.features, header=True)
        if 'name' not in self.features.columns:
            raise ValueError('Features dataframe must include name column')

    def _load_labels(self) -> None:
        """
        Load labels to train ML model on
        """
        self.labels = self.spark.read.csv(self.data + self.file_names.labels, header=True)
        if 'name' not in self.labels.columns:
            raise ValueError('Labels dataframe must include name column')
        if 'label' not in self.labels.columns:
            raise ValueError('Labels dataframe must include label column')
        if 'weight' not in self.labels.columns:
            self.labels = self.labels.withColumn('weight', lit(1))
        self.labels = self.labels.select(['name', 'label', 'weight'])

    def _load_targeting(self) -> None:
        """
        Load targeting data.
        """
        self.targeting = pd.read_csv(self.data + self.file_names.targeting)
        self.targeting['random'] = np.random.rand(len(self.targeting))

        # TODO: use decorator
        # Unweighted data
        self.unweighted_targeting = self.targeting.copy()
        self.unweighted_targeting['weight'] = 1

        # Weighted data
        self.weighted_targeting = self.targeting.copy()
        if 'weight' not in self.weighted_targeting.columns:
            self.weighted_targeting['weight'] = 1
        else:
            self.weighted_targeting['weight'] = (self.weighted_targeting['weight'] /
                                                 self.weighted_targeting['weight'].min())
        self.weighted_targeting = pd.DataFrame(np.repeat(self.weighted_targeting.values,
                                                         self.weighted_targeting['weight'],
                                                         axis=0),
                                               columns=self.weighted_targeting.columns) \
            .astype(self.unweighted_targeting.dtypes)

    def _load_fairness(self) -> None:
        """
        Load fairness data.
        """
        self.fairness = pd.read_csv(self.data + self.file_names.fairness)
        self.fairness['random'] = np.random.rand(len(self.fairness))

        # TODO: use decorator
        # Unweighted data
        self.unweighted_fairness = self.fairness.copy()
        self.unweighted_fairness['weight'] = 1

        # Weighted data
        self.weighted_fairness = self.fairness.copy()
        if 'weight' not in self.weighted_fairness.columns:
            self.weighted_fairness['weight'] = 1
        else:
            self.weighted_fairness['weight'] = (self.weighted_fairness['weight'] /
                                                 self.weighted_fairness['weight'].min())
        self.weighted_fairness = pd.DataFrame(np.repeat(self.weighted_fairness.values,
                                                    self.weighted_fairness['weight'],
                                                    axis=0),
                                          columns=self.weighted_fairness.columns) \
            .astype(self.unweighted_fairness.dtypes)

    def _load_wealth_map(self) -> None:
        # Load wealth/income map
        if self.file_names.rwi:
            self.rwi = pd.read_csv(self.data + self.file_names.rwi, dtype={'quadkey': str})
        else:
            raise ValueError("Missing path to wealth map in config file.")

    def _load_survey(self, dataframe: Optional[PandasDataFrame] = None) -> None:
        # Load survey data from disk if dataframe not available
        if dataframe is not None:
            self.survey_data = dataframe
        elif self.file_names.survey is not None:
            self.survey_data = pd.read_csv(self.data + self.file_names.survey)
        else:
            raise ValueError("Missing path to survey data in config file.")
        # Add weights column if missing
        if 'weight' not in self.survey_data.columns:
            self.survey_data['weight'] = 1

    def merge(self) -> None:
        """
        Merge features and labels, split into x and y dataframes
        """
        if self.features is None or self.labels is None:
            raise ValueError("Features and/or labels have not been loaded!")
        print('Number of observations with features: %i (%i unique)' %
              (self.features.count(), self.features.select('name').distinct().count()))
        print('Number of observations with labels: %i (%i unique)' %
              (self.labels.count(), self.labels.select('name').distinct().count()))

        merged = self.labels.join(self.features, on='name', how='inner')
        print('Number of matched observations: %i (%i unique)' %
              (merged.count(), merged.select('name').distinct().count()))

        save_df(merged, self.outputs + '/merged.csv')
        self.merged = pd.read_csv(self.outputs + '/merged.csv')
        self.x = self.merged.drop(['name', 'label', 'weight'], axis=1)
        self.y = self.merged['label']
        # Make the smallest weight 1
        self.weights = self.merged['weight'] / self.merged['weight'].min()

    def load_data(self, data_type_map: Mapping[DataType, Optional[Union[SparkDataFrame, PandasDataFrame]]]) -> None:
        """
        Load all datasets defined by data_type_map; raise an error if any of them failed to load
        Args:
            data_type_map: mapping between DataType(s) and dataframes, if provided. If None look at config file
        """
        # Iterate through provided dtypes and load respective datasets
        for key, value in data_type_map.items():
            fn = self.data_type_to_fn_map[key]
            if 'dataframe' in inspect.getfullargspec(fn).args:
                fn(dataframe=value)
            else:
                fn()

        # Check if any datasets failed to load, raise an error if true
        failed_load = []
        for key in data_type_map:
            dataset = key.name.lower()
            if getattr(self, dataset) is None:
                failed_load.append(dataset)
        if failed_load:
            raise ValueError(f"The following datasets failed to load: {', '.join(failed_load)}")

    def filter_dates(self, start_date: str, end_date: str) -> None:
        """
        Filter data outside [start_date, end_date] (inclusive) in all available datasets
        Args:
            start_date: e.g. '2020-01-01'
            end_date: e.g. '2020-01-10'
        """
        for dataset_name in self.datasets:
            dataset = getattr(self, '_' + dataset_name, None)
            if dataset is not None:
                setattr(self, dataset_name, filter_dates_dataframe(dataset, start_date, end_date))

    def deduplicate(self) -> None:
        """
        Remove duplicate rows from alla available datasets
        """
        for dataset_name in self.datasets:
            dataset = getattr(self, '_' + dataset_name, None)
            if dataset is not None:
                setattr(self, dataset_name, dataset.distinct())

    def remove_spammers(self, spammer_threshold: float = 100) -> List[str]:
        # Raise exception if no CDR, since spammers are calculated only on the basis of call and text
        if self.cdr is None:
            raise ValueError('CDR must be loaded to identify and remove spammers.')

        # Get average number of calls and SMS per day
        grouped = (self.cdr
                   .groupby('caller_id', 'txn_type')
                   .agg(count(lit(0)).alias('n_transactions'),
                        countDistinct(col('day')).alias('active_days'))
                   .withColumn('count', col('n_transactions') / col('active_days')))

        # Get list of spammers
        self.spammers = grouped.where(col('count') > spammer_threshold).select('caller_id').distinct().rdd.map(
            lambda r: r[0]).collect()
        pd.DataFrame(self.spammers).to_csv(self.outputs + 'datasets/spammers.csv', index=False)
        print('Number of spammers identified: %i' % len(self.spammers))

        # Remove transactions (incoming or outgoing) associated with spammers from all dataframes
        self.cdr = self.cdr.where(~col('caller_id').isin(self.spammers))
        self.cdr = self.cdr.where(~col('recipient_id').isin(self.spammers))
        if self.recharges is not None:
            self.recharges = self.recharges.where(~col('caller_id').isin(self.spammers))
        if self.mobiledata is not None:
            self.mobiledata = self.mobiledata.where(~col('caller_id').isin(self.spammers))
        if self.mobilemoney is not None:
            self.mobilemoney = self.mobilemoney.where(~col('caller_id').isin(self.spammers))
            self.mobilemoney = self.mobilemoney.where(~col('recipient_id').isin(self.spammers))

        return self.spammers

    def filter_outlier_days(self, num_sds: float = 2) -> List:
        # Raise exception if no CDR, since spammers are calculated only on the basis of call and text
        if self.cdr is None:
            raise ValueError('CDR must be loaded to identify and remove outlier days.')

        # If haven't already obtained timeseries of subscribers by day (e.g. in diagnostic plots), calculate it
        if not os.path.isfile(self.outputs + 'datasets/CDR_transactionsbyday.csv'):
            save_df(self.cdr.groupby(['txn_type', 'day']).count(), self.outputs + 'datasets/CDR_transactionsbyday.csv')

        # Read in timeseries of subscribers by day
        timeseries = pd.read_csv(self.outputs + 'datasets/CDR_transactionsbyday.csv')

        # Calculate timeseries of all transaction (voice + SMS together)
        timeseries = timeseries.groupby('day', as_index=False).agg('sum')

        # Calculate top and bottom acceptable values
        bottomrange = timeseries['count'].mean() - num_sds * timeseries['count'].std()
        toprange = timeseries['count'].mean() + num_sds * timeseries['count'].std()

        # Obtain list of outlier days
        outliers = timeseries[(timeseries['count'] < bottomrange) | (timeseries['count'] > toprange)]
        outliers.to_csv(self.outputs + 'datasets/outlier_days.csv', index=False)
        outliers = list(outliers['day'])
        outliers = [outlier.split('T')[0] for outlier in outliers]
        print('Outliers removed: ' + ', '.join(outliers))

        # Remove outlier days from all datasets
        for df_name in ['cdr', 'recharges', 'mobiledata', 'mobilemoney']:
            for outlier in outliers:
                outlier = pd.to_datetime(outlier)
                if getattr(self, df_name, None) is not None:
                    setattr(self, df_name, getattr(self, df_name)
                            .where((col('timestamp') < outlier) |
                                   (col('timestamp') >= outlier + pd.Timedelta(value=1, unit='days'))))

        return outliers

    def remove_survey_outliers(self, cols: List[str], num_sds: float = 2., dry_run: bool = False) -> None:
        """
        Removes observations with outliers in the columns listed in 'cols' from the survey data.

        Args:
            cols: Columns used to identify outliers.
            num_sds: Number of standard deviations used to identify outliers.
            dry_run: If True, only prints the number of outliers without removing them.
        """
        # Raise exception if survey data has not been loaded
        if self.survey_data is None:
            raise ValueError('Survey data must be loaded to identify and remove outliers.')

        data = self.survey_data.set_index('unique_id')[cols]

        # Calculate top and bottom acceptable values
        bottomrange = data.mean() - num_sds * data.std()
        toprange = data.mean() + num_sds * data.std()

        outliers = []
        for i, (col, bottom) in enumerate(bottomrange.iteritems()):
            outliers = outliers + list(data[(data[col] < bottom) | (data[col] > toprange[0])].index.values)
        outliers = set(outliers)

        if dry_run:
            print(f"There are {len(outliers)} outliers that could be removed.")
        else:
            self.survey_data = self.survey_data[~self.survey_data['unique_id'].isin(outliers)]
            print(f"Removed {len(outliers)} outliers!")


class OptDataStore(DataStore):
    def __init__(self, cfg_dir: str):
        super(OptDataStore, self).__init__(cfg_dir)
        self._user_consent: SparkDataFrame

    @property
    def user_consent(self) -> SparkDataFrame:
        return self._user_consent

    @user_consent.setter
    def user_consent(self, val: SparkDataFrame) -> None:
        """
        Whenever the user consent table is updated, also update all datasets in the datastore to include only users that
        have given their consent
        Args:
            val: new user consent table as spark df
        """
        self._user_consent = val
        # Get name of user id column
        user_col_name = val.columns[0] if val is not None else None
        # Iterate through all private datasets, update the corresponding visible datasets
        for dataset_name in self.datasets:
            dataset = getattr(self, '_' + dataset_name, None)
            if dataset is not None and val is not None:
                setattr(self, dataset_name, dataset.join(val.where(col('include') == True).select(user_col_name),
                                                         on=user_col_name, how='inner'))

    def initialize_user_consent_table(self) -> None:
        """
        Create table of all user ids present in the datasets, and whether they should be included in the analysis or not
        This is defined by the opt_in_default parameter specified in the config file
        """
        # Create internal use datasets which do not get updated based on the consent table
        for dataset_name in self.datasets:
            if getattr(self, dataset_name, None) is not None:
                setattr(self, '_' + dataset_name, getattr(self, dataset_name))
        # Get all available datasets and create consent table
        data = []
        for dataset_name in self.datasets:
            if getattr(self, '_' + dataset_name, None) is not None:
                data.append(getattr(self, '_' + dataset_name))
        if 'subscriber_id' in data[0].columns:  # home location
            user_col_name = 'subscriber_id'
        elif 'caller_id' in data[0].columns:  # featurizer
            user_col_name = 'caller_id'
        else:  # ml
            user_col_name = 'name'
        self.user_consent = generate_user_consent_list(data, user_id_col=user_col_name,
                                                       opt_in=self.cfg.params.opt_in_default)

        # Check if a user consent file has been provided, and if so set consent flags appropriately
        if self.file_names.user_consent is not None:
            user_consent_df = pd.read_csv(self.data + self.file_names.user_consent)
            if 'user_id' not in user_consent_df.columns:
                raise ValueError("The user consent table should have a 'user_id' column")
            # If there's just a user id column, set those user ids' consent to the opposite of opt_in_default
            if len(user_consent_df.columns) == 1:
                user_ids = list(user_consent_df['user_id'])
                if self.cfg.params.opt_in_default:
                    self.opt_out(user_ids=user_ids)
                else:
                    self.opt_in(user_ids=user_ids)
            elif len(user_consent_df.columns) == 2:
                if 'include' not in user_consent_df.columns or user_consent_df['include'].dtype != bool:
                    raise ValueError("The consent column should be called 'include' and have True/False values")
                user_ids_in = list(user_consent_df[user_consent_df['include'] == True]['user_id'])
                user_ids_out = list(user_consent_df[user_consent_df['include'] == False]['user_id'])
                self.opt_in(user_ids=user_ids_in)
                self.opt_out(user_ids=user_ids_out)
            else:
                raise ValueError("The user consent table should have at most two columns, one for the user id and "
                                 "another for the consent flag")

    def opt_in(self, user_ids: List[str]) -> None:
        """
        Update the user consent table based on list of user ids that have opted in
        Args:
            user_ids: list of user ids to flag as opted in, i.e. include = True
        """
        user_col_name = self.user_consent.columns[0]
        self.user_consent = (self.user_consent
                             .withColumn('include', F.when(col(user_col_name).isin(user_ids), True)
                                         .otherwise(col('include'))))

    def opt_out(self, user_ids: List[str]) -> None:
        """
        Update the user consent table based on list of user ids that have opted out
        Args:
            user_ids: list of user ids to flag as opted out, i.e. include = False
        """
        user_col_name = self.user_consent.columns[0]
        self.user_consent = (self.user_consent
                             .withColumn('include', F.when(col(user_col_name).isin(user_ids), False)
                                         .otherwise(col('include'))))
