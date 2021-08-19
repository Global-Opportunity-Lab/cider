from abc import ABC, abstractmethod
from box import Box
from collections import defaultdict
from helpers.io_utils import *
from helpers.opt_utils import *
import pyspark.sql.functions as F
from pyspark.sql.functions import col
import yaml


class InitializerInterface(ABC):
    @abstractmethod
    def load_cdr(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_antennas(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_shapefiles(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_home_ground_truth(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_poverty_scores(self, *args, **kwargs):
        pass


class DataStore(InitializerInterface):
    def __init__(self, cfg_dir: str):
        # Read config file
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
        for dataset_name in self.datasets:
            setattr(self, dataset_name, None)
        self.antennas = None
        self.shapefiles = {}
        self.ground_truth = None
        self.poverty_scores = None

    def load_cdr(self, dataframe=None):
        fpath = self.data + self.file_names.cdr if self.file_names.cdr is not None else None
        if self.file_names.cdr is not None or dataframe is not None:
            print('Loading CDR...')
            cdr = load_cdr(self.cfg, fpath, df=dataframe)
            self.cdr = cdr
            self.cdr_bandicoot = None

    def load_antennas(self, dataframe=None):
        fpath = self.data + self.file_names.antennas if self.file_names.antennas is not None else None
        if self.file_names.antennas is not None or dataframe is not None:
            print('Loading antennas...')
            self.antennas = load_antennas(self.cfg, fpath, df=dataframe)

    def load_shapefiles(self):
        # Load shapefiles
        shapefiles = self.file_names.shapefiles
        for shapefile_fname in shapefiles.keys():
            self.shapefiles[shapefile_fname] = load_shapefile(self.data + shapefiles[shapefile_fname])

    def load_home_ground_truth(self):
        if self.file_names.home_ground_truth is not None:
            self.ground_truth = pd.read_csv(self.data + self.file_names.home_ground_truth)
        else:
            print('No ground truth data for home locations has been specified.')

    def load_poverty_scores(self):
        if self.file_names.poverty_scores is not None:
            self.poverty_scores = pd.read_csv(self.data + self.file_names.poverty_scores)

    def load_data(self, module, dataframes=None):
        dataframes = dataframes if dataframes else defaultdict(lambda: None)
        cdr_df = dataframes['cdr']
        antennas_df = dataframes['antennas']
        if module == 'home_location':
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

    def filter_dates(self, start_date, end_date):
        for dataset_name in self.datasets:
            dataset = getattr(self, '_' + dataset_name, None)
            if dataset is not None:
                setattr(self, dataset_name, filter_dates_dataframe(dataset, start_date, end_date))

    def deduplicate(self):
        for dataset_name in self.datasets:
            dataset = getattr(self, '_' + dataset_name, None)
            if dataset is not None:
                setattr(self, dataset_name, dataset.distinct())


class OptDataStore(DataStore):
    def __init__(self, cfg_dir: str):
        super(OptDataStore, self).__init__(cfg_dir)

        self._user_consent = None

    @property
    def user_consent(self):
        return self._user_consent

    @user_consent.setter
    def user_consent(self, val):
        self._user_consent = val
        user_col_name = val.columns[0] if val is not None else None
        for dataset_name in self.datasets:
            dataset = getattr(self, '_' + dataset_name, None)
            if dataset is not None and val is not None:
                setattr(self, dataset_name, dataset.join(val.where(col('include') == True).select(user_col_name),
                                                         on=user_col_name, how='inner'))

    def initialize_user_consent_table(self):
        for dataset_name in self.datasets:
            if getattr(self, dataset_name, None) is not None:
                setattr(self, '_' + dataset_name, getattr(self, dataset_name))
        data = []
        for dataset_name in self.datasets:
            if getattr(self, '_' + dataset_name, None) is not None:
                data.append(getattr(self, '_' + dataset_name))
        user_col_name = 'subscriber_id' if 'subscriber_id' in data[0].columns else 'caller_id'
        self.user_consent = generate_user_consent_list(data, user_id_col=user_col_name,
                                                       opt_in=self.cfg.params.opt_in_default)

    def opt_in(self, user_ids):
        user_col_name = self.user_consent.columns[0]
        self.user_consent = (self.user_consent
                             .withColumn('include', F.when(col(user_col_name).isin(user_ids), True)
                                                     .otherwise(col('include'))))

    def opt_out(self, user_ids):
        user_col_name = self.user_consent.columns[0]
        self.user_consent = (self.user_consent
                             .withColumn('include', F.when(col(user_col_name).isin(user_ids), False)
                                                     .otherwise(col('include'))))
