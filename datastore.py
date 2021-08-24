from abc import ABC, abstractmethod
from box import Box
from collections import defaultdict
from helpers.io_utils import *
from helpers.opt_utils import *
from pandas import DataFrame as PandasDataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import col, count, lit
from typing import Dict, Union
import yaml


class InitializerInterface(ABC):
    @abstractmethod
    def load_cdr(self, dataframe):
        pass

    @abstractmethod
    def load_antennas(self, dataframe):
        pass

    @abstractmethod
    def load_recharges(self, dataframe):
        pass

    @abstractmethod
    def load_mobiledata(self, dataframe):
        pass

    @abstractmethod
    def load_mobilemoney(self, dataframe):
        pass

    @abstractmethod
    def load_shapefiles(self):
        pass

    @abstractmethod
    def load_home_ground_truth(self):
        pass

    @abstractmethod
    def load_poverty_scores(self):
        pass


class DataStore(InitializerInterface):
    def __init__(self, cfg_dir: str):
        # Read config file and store paths
        with open(cfg_dir, "r") as ymlfile:
            cfg = Box(yaml.load(ymlfile, Loader=yaml.FullLoader))
        self.cfg = cfg
        data = cfg.path.data
        self.data = data
        outputs = cfg.path.outputs
        self.outputs = outputs
        file_names = cfg.path.file_names
        self.file_names = file_names

        # Parameters
        self.filter_hours = self.cfg.params.home_location.filter_hours
        self.geo = self.cfg.col_names.geo

        # Spark setup
        spark = get_spark_session(cfg)
        self.spark = spark

        # Possible datasets
        self.datasets = ['cdr', 'recharges', 'mobiledata', 'mobilemoney']
        self.cdr = None
        self.cdr_bandicoot = None
        self.recharges = None
        self.mobiledata = None
        self.mobilemoney = None
        self.antennas = None
        self.shapefiles = {}
        self.ground_truth = None
        self.poverty_scores = None

    def load_cdr(self, dataframe: Union[SparkDataFrame, PandasDataFrame] = None) -> None:
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

    def load_antennas(self, dataframe: Union[SparkDataFrame, PandasDataFrame] = None) -> None:
        fpath = self.data + self.file_names.antennas if self.file_names.antennas is not None else None
        if fpath or dataframe is not None:
            print('Loading antennas...')
            self.antennas = load_antennas(self.cfg, fpath, df=dataframe)

    def load_recharges(self, dataframe: Union[SparkDataFrame, PandasDataFrame] = None) -> None:
        """
        Load recharges data: use file path specified in config as default, or spark/pandas df
        Args:
            dataframe: spark/pandas df to assign if available
        """
        fpath = self.data + self.file_names.recharges if self.file_names.recharges is not None else None
        if fpath or dataframe is not None:
            print('Loading recharges...')
            self.recharges = load_recharges(self.cfg, fpath, df=dataframe)

    def load_mobiledata(self, dataframe: Union[SparkDataFrame, PandasDataFrame] = None) -> None:
        """
        Load mobile data: use file path specified in config as default, or spark/pandas df
        Args:
            dataframe: spark/pandas df to assign if available
        """
        fpath = self.data + self.file_names.mobiledata if self.file_names.mobiledata is not None else None
        if fpath or dataframe is not None:
            print('Loading mobile data...')
            self.mobiledata = load_mobiledata(self.cfg, fpath, df=dataframe)

    def load_mobilemoney(self, dataframe: Union[SparkDataFrame, PandasDataFrame] = None) -> None:
        """
        Load mobile money data: use file path specified in config as default, or spark/pandas df
        Args:
            dataframe: spark/pandas df to assign if available
        """
        fpath = self.data + self.file_names.mobilemoney if self.file_names.mobilemoney is not None else None
        if fpath or dataframe is not None:
            print('Loading mobile data...')
            self.mobilemoney = load_mobilemoney(self.cfg, fpath, df=dataframe)

    def load_shapefiles(self) -> None:
        """
        Iterate through shapefiles specified in config and load them in self.shapefiles dictionary
        """
        # Load shapefiles
        shapefiles = self.file_names.shapefiles
        for shapefile_fname in shapefiles.keys():
            self.shapefiles[shapefile_fname] = load_shapefile(self.data + shapefiles[shapefile_fname])

    def load_home_ground_truth(self) -> None:
        """
        Load ground truth data for home locations
        """
        if self.file_names.home_ground_truth is not None:
            self.ground_truth = pd.read_csv(self.data + self.file_names.home_ground_truth)
        else:
            print('No ground truth data for home locations has been specified.')

    def load_poverty_scores(self) -> None:
        """
        Load poverty scores (e.g. those produced by the ML module)
        """
        if self.file_names.poverty_scores is not None:
            self.poverty_scores = pd.read_csv(self.data + self.file_names.poverty_scores)

    def load_data(self, module: str, dataframes:  Dict[str, Union[SparkDataFrame, PandasDataFrame]] = None) -> None:
        """
        Load and process all datasets required by a module
        Args:
            module: module to load data for
            dataframes: dictionary of dataframes to use; if none is passed reads from file paths in config
        """
        # Create defaultdict to avoid keyerrors
        dataframes = dataframes if dataframes else defaultdict(lambda: None)
        cdr_df = dataframes['cdr']
        recharges_df = dataframes['recharges']
        mobiledata_df = dataframes['mobiledata']
        mobilemoney_df = dataframes['mobilemoney']
        antennas_df = dataframes['antennas']
        if module == 'featurizer':
            self.load_cdr(dataframe=cdr_df)
            self.load_recharges(dataframe=recharges_df)
            self.load_mobiledata(dataframe=mobiledata_df)
            self.load_mobilemoney(dataframe=mobilemoney_df)
            self.load_antennas(dataframe=antennas_df)
            self.load_shapefiles()
        elif module == 'home_location':
            self.load_cdr(dataframe=cdr_df)
            self.load_antennas(dataframe=antennas_df)
            self.load_shapefiles()
            # Clean and merge CDR data
            outgoing = (self.cdr
                        .select(['caller_id', 'caller_antenna', 'timestamp', 'day'])
                        .withColumnRenamed('caller_id', 'subscriber_id')
                        .withColumnRenamed('caller_antenna', 'antenna_id'))
            incoming = (self.cdr
                        .select(['recipient_id', 'recipient_antenna', 'timestamp', 'day'])
                        .withColumnRenamed('recipient_id', 'subscriber_id')
                        .withColumnRenamed('recipient_antenna', 'antenna_id'))
            self.cdr = (outgoing
                        .select(incoming.columns)
                        .union(incoming)
                        .na.drop()
                        .withColumn('hour', hour('timestamp')))

            # Filter CDR to only desired hours
            if self.filter_hours is not None:
                self.cdr = self.cdr.where(col('hour').isin(self.filter_hours))

            # Get tower ID for each transaction
            if self.geo == 'tower_id':
                self.cdr = (self.cdr
                            .join(self.antennas
                                  .select(['antenna_id', 'tower_id']).na.drop(), on='antenna_id', how='inner'))

            # Get polygon for each transaction based on antenna latitude and longitudes
            elif self.geo in self.shapefiles.keys():
                antennas = self.antennas.na.drop().toPandas()
                antennas = gpd.GeoDataFrame(antennas,
                                            geometry=gpd.points_from_xy(antennas['longitude'],
                                                                        antennas['latitude']))
                antennas.crs = {"init": "epsg:4326"}
                antennas = gpd.sjoin(antennas, self.shapefiles[self.geo], op='within', how='left')[
                    ['antenna_id', 'region']].rename({'region': self.geo}, axis=1)
                antennas = self.spark.createDataFrame(antennas).na.drop()
                length_before = self.cdr.count()
                self.cdr = self.cdr.join(antennas, on='antenna_id', how='inner')
                length_after = self.cdr.count()
                if length_before != length_after:
                    print('Warning: %i (%.2f percent of) transactions not located in a polygon' %
                          (length_before - length_after, 100 * (length_before - length_after) / length_before))

            elif self.geo != 'antenna_id':
                raise ValueError('Invalid geography, must be antenna_id, tower_id, or shapefile name')
        else:
            raise ValueError(f"The module name provided - '{module}' - is incorrect or not supported.")

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

    def remove_spammers(self, spammer_threshold=100):
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
        pd.DataFrame(self.spammers).to_csv(self.outputs + '/datasets/spammers.csv', index=False)
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

    def filter_outlier_days(self, num_sds=2):
        # Raise exception if no CDR, since spammers are calculated only on the basis of call and text
        if self.cdr is None:
            raise ValueError('CDR must be loaded to identify and remove outlier days.')

        # If haven't already obtained timeseries of subscribers by day (e.g. in diagnostic plots), calculate it
        if not os.path.isfile(self.outputs + '/datasets/CDR_transactionsbyday.csv'):
            save_df(self.cdr.groupby(['txn_type', 'day']).count(), self.outputs + '/datasets/CDR_transactionsbyday.csv')

        # Read in timeseries of subscribers by day
        timeseries = pd.read_csv(self.outputs + '/datasets/CDR_transactionsbyday.csv')

        # Calculate timeseries of all transaction (voice + SMS together)
        timeseries = timeseries.groupby('day', as_index=False).agg('sum')

        # Calculate top and bottom acceptable values
        bottomrange = timeseries['count'].mean() - num_sds * timeseries['count'].std()
        toprange = timeseries['count'].mean() + num_sds * timeseries['count'].std()

        # Obtain list of outlier days
        outliers = timeseries[(timeseries['count'] < bottomrange) | (timeseries['count'] > toprange)]
        outliers.to_csv(self.outputs + '/datasets/outlier_days.csv', index=False)
        outliers = list(outliers['day'])
        print('Outliers removed: ' + ', '.join([outlier.split('T')[0] for outlier in outliers]))

        # Remove outlier days from all datasets
        for df_name in ['cdr', 'recharges', 'mobiledata', 'mobilemoney']:
            for outlier in outliers:
                outlier = pd.to_datetime(outlier)
                if getattr(self, df_name, None) is not None:
                    setattr(self, df_name, getattr(self, df_name)
                            .where((col('timestamp') < outlier) | (col('timestamp') >= outlier + pd.Timedelta(days=1))))

        return outliers


class OptDataStore(DataStore):
    def __init__(self, cfg_dir: str):
        super(OptDataStore, self).__init__(cfg_dir)
        self._user_consent = None

    @property
    def user_consent(self):
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
        user_col_name = 'subscriber_id' if 'subscriber_id' in data[0].columns else 'caller_id'
        self.user_consent = generate_user_consent_list(data, user_id_col=user_col_name,
                                                       opt_in=self.cfg.params.opt_in_default)

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

    def opt_out(self, user_ids: List[str]):
        """
        Update the user consent table based on list of user ids that have opted out
        Args:
            user_ids: list of user ids to flag as opted out, i.e. include = False
        """
        user_col_name = self.user_consent.columns[0]
        self.user_consent = (self.user_consent
                             .withColumn('include', F.when(col(user_col_name).isin(user_ids), False)
                                                     .otherwise(col('include'))))
