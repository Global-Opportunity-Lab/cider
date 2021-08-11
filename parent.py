from box import Box
from helpers.io_utils import *
import pyspark.sql.functions as F
from pyspark.sql.functions import col
import yaml


class Parent:
    def __init__(self, cfg_dir: str, module: str, clean_folders: bool = False):
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


class Opt(Parent):
    def __init__(self, cfg_dir: str, module: str, clean_folders: bool = False):
        super(Opt, self).__init__(cfg_dir, module, clean_folders)

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
