from box import Box
from helpers.io_utils import *
from helpers.opt_utils import *
import pyspark.sql.functions as F
from pyspark.sql.functions import col
import yaml


class Initializer:
    def __init__(self, cfg_dir: str, module: str, dataframes: dict = None, clean_folders: bool = False):
        # Read config file
        with open(cfg_dir, "r") as ymlfile:
            cfg = Box(yaml.load(ymlfile, Loader=yaml.FullLoader))
        self.cfg = cfg
        data = cfg.path[module].data
        self.data = data
        outputs = cfg.path[module].outputs
        self.outputs = outputs
        file_names = cfg.path[module].file_names
        self.file_names = file_names

        # Prepare working directories
        make_dir(outputs, clean_folders)
        make_dir(outputs + '/outputs/')
        make_dir(outputs + '/maps/')
        make_dir(outputs + '/tables/')

        # Spark setup
        spark = get_spark_session(cfg)
        self.spark = spark

        # Possible datasets
        self.datasets = {'cdr': None, 'recharges': None, 'mobiledata': None, 'mobilemoney': None}

        # Load CDR data
        dataframe = dataframes['cdr'] if dataframes is not None and 'cdr' in dataframes.keys() else None
        fpath = data + file_names.cdr if file_names.cdr is not None else None
        if file_names.cdr is not None or dataframe is not None:
            print('Loading CDR...')
            self.cdr_full = load_cdr(self.cfg, fpath, df=dataframe)
            self.cdr_bandicoot = None
        else:
            self.cdr_full = None

        # Load antennas data
        dataframe = dataframes['antennas'] if dataframes is not None and 'antennas' in dataframes.keys() else None
        fpath = data + file_names.antennas if file_names.antennas is not None else None
        if file_names.antennas is not None or dataframe is not None:
            print('Loading antennas...')
            self.antennas = load_antennas(self.cfg, fpath, df=dataframe)
        else:
            self.antennas = None


class Opt(Initializer):
    def __init__(self, cfg_dir: str, module: str, dataframes: dict = None, clean_folders: bool = False):
        super(Opt, self).__init__(cfg_dir, module, dataframes, clean_folders)

    @property
    def user_consent(self):
        return self._user_consent

    @user_consent.setter
    def user_consent(self, val):
        self._user_consent = val
        for dataset_name in self.datasets:
            dataset = getattr(self, dataset_name + '_full', None)
            if dataset is not None:
                setattr(self, dataset_name, dataset.join(val.where(col('include') == True).select(self.user_id),
                                                         on=self.user_id, how='inner'))

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
