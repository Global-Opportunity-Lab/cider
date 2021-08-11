from box import Box
from helpers.io_utils import *
import pandas as pd
import yaml


class Parent:
    def __init__(self, cfg_dir, module, clean_folders=False):
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

        # Prepare working directory
        make_dir(outputs, clean_folders)
        make_dir(outputs + '/outputs/')
        make_dir(outputs + '/maps/')
        make_dir(outputs + '/tables/')

        # Spark setup
        spark = get_spark_session(cfg)
        self.spark = spark
