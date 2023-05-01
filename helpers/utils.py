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

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from box import Box
from numpy import ndarray
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format, lit
from pyspark.sql.types import IntegerType, StringType
from typing_extensions import Literal
from yaml import FullLoader, load as yaml_load


def get_spark_session(cfg: Box) -> SparkSession:
    """
    Gets or creates spark session, with context and logging preferences set
    """
    # Build spark session
    
    # Recursively get all specified spark options
    def all_spark_options(config: Box):
        for key, value in config.items():
            if value is None:
                return None

            elif isinstance(value, Box):
                for r_key, r_value in all_spark_options(value):
                    yield f'{key}.{r_key}', r_value
        
            else:
                yield key, value

    spark_session_builder = SparkSession.builder.config("spark.executorEnv.SPARK_LOG_DIR", "/home/nanoloans/spark_logs")
    for spark_option, value in all_spark_options(cfg.spark):
        spark_session_builder = spark_session_builder.config(f'spark.{spark_option}', value)
    
    # Cider config used to expect some Spark config specified a little differently - check for those entries
    # for backwards compatibility. Throw warnings because this config file can't be understood using Spark
    # documentation.
    if 'app_name' in cfg.spark:
        spark_session_builder = spark_session_builder.appName(cfg.spark.app_name)
        warnings.warn('Please specify app name using spark: app: name rather than spark: app_name.')
        
    if ('files' in cfg.spark) and ('max_partition_bytes' in cfg.spark.files):
        spark_session_builder = spark_session_builder.config("spark.sql.files.maxPartitionBytes", cfg.spark.files.max_partition_bytes) 
        warnings.warn(
            'Please specify max bytes per partition using variable name(s) specified at '
            'https://spark.apache.org/docs/latest/configuration.html#available-properties'
        )
        
    if ('driver' in cfg.spark) and ('max_result_size' in cfg.spark.driver):
        spark_session_builder = (
            spark_session_builder.config("spark.driver.maxResultSize", cfg.spark.driver.max_result_size)
        )
        warnings.warn(
            'Please specify max result size using variable name(s) specified at '
            'https://spark.apache.org/docs/latest/configuration.html#available-properties'
        )

    # Create the Spark session
    spark = spark_session_builder.getOrCreate()
    spark.sparkContext.setLogLevel(cfg.spark.loglevel)
    
    return spark


def save_df(df: SparkDataFrame, out_file_path: Path, sep: str = ',') -> None:
    """
    Saves spark dataframe to csv file.
    """
    # we need to work around spark's automatic partitioning/naming

    # create a temporary folder in the directory where the output will ultimately live
    temp_folder = out_file_path.parent / 'temp'
    
    # Ask spark to write output there. The repartition(1) call will tell spark to write a single file.
    # It will name it with some meaningless partition name, but we can find it easily bc it's the only
    # csv in the temp directory.
    df.repartition(1).write.csv(path=str(temp_folder), mode="overwrite", header="true", sep=sep)
    spark_generated_file_name = [
        fname for fname in os.listdir(temp_folder) if os.path.splitext(fname)[1] == '.csv'
    ][0]
    
    # move the file out of the temporary directory and rename it
    os.rename(temp_folder / spark_generated_file_name, out_file_path)
    
    # delete the temp directory and everything in it
    shutil.rmtree(temp_folder)

def read_csv(spark_session, file_path: Path, **kwargs):
    """
    A wrapper around spark.read.csv which accepts pathlib.Path objects as input.
    """
    return spark_session.read.csv(str(file_path), **kwargs)


def save_parquet(df: SparkDataFrame, out_file_path: Path) -> None:
    """
    Save spark dataframe to parquet file
    """
    df.write.mode('overwrite').parquet(str(out_file_path))


def filter_dates_dataframe(df: SparkDataFrame,
                           start_date: str, end_date: str, colname: str = 'timestamp') -> SparkDataFrame:
    """
    Filter dataframe rows whose timestamp is outside [start_date, end_date)

    Args:
        df: spark df
        start_date: initial date to keep
        end_date: first date to exclude
        colname: name of timestamp column

    Returns: filtered spark df

    """
    if colname not in df.columns:
        raise ValueError('Cannot filter dates because missing timestamp column')
    df = df.where(col(colname) >= pd.to_datetime(start_date))
    df = df.where(col(colname) < pd.to_datetime(end_date) + pd.Timedelta(value=1, unit='days'))
    return df


def make_dir(directory_path: Path, remove: bool = False) -> None:
    """
    Create new directory

    Args:
        directory_path: directory path
        remove: whether to replace the directory with an empty one if it's already present
    """
    if directory_path.is_dir() and remove:
        shutil.rmtree(directory_path)

    directory_path.mkdir(parents=True, exist_ok=True)


def flatten_lst(lst: List[List]) -> List:
    return [item for sublist in lst for item in sublist]


def flatten_folder(args: Tuple) -> List[str]:
    ids, recs_folder = args
    unmatched: List[str] = []
    for p in ids:
        try:
            fname = 'name=' + p
            os.system('mv ' + recs_folder + '/' + fname + '/*.csv ' + recs_folder + '/' + p + '.csv')
        except:
            unmatched = unmatched + [p]
    return unmatched


def cdr_bandicoot_format(cdr: SparkDataFrame, antennas: SparkDataFrame, cfg: Box) -> SparkDataFrame:
    """
    Convert CDR df into format that can be used by bandicoot

    Args:
        cdr: spark df with CDRs
        antennas: antenna dataframe
        cfg: box object with cdr column names

    Returns: spark df in bandicoot format
    """

    cols = list(cfg.keys())

    outgoing = cdr.select(cols)\
        .withColumnRenamed('txn_type', 'interaction')\
        .withColumnRenamed('caller_id', 'name')\
        .withColumnRenamed('recipient_id', 'correspondent_id')\
        .withColumnRenamed('timestamp', 'datetime')\
        .withColumnRenamed('duration', 'call_duration')\
        .withColumnRenamed('caller_antenna', 'antenna_id')\
        .withColumn('direction', lit('out'))\
        .drop('recipient_antenna')

    incoming = cdr.select(cols)\
        .withColumnRenamed('txn_type', 'interaction')\
        .withColumnRenamed('recipient_id', 'name')\
        .withColumnRenamed('caller_id', 'correspondent_id')\
        .withColumnRenamed('timestamp', 'datetime')\
        .withColumnRenamed('duration', 'call_duration')\
        .withColumnRenamed('recipient_antenna', 'antenna_id')\
        .withColumn('direction', lit('in'))\
        .drop('caller_antenna')

    cdr_bandicoot = outgoing.select(incoming.columns).union(incoming)\
        .withColumn('call_duration', col('call_duration').cast(IntegerType()).cast(StringType()))\
        .withColumn('datetime', date_format(col('datetime'), 'yyyy-MM-dd HH:mm:ss'))
    
    if antennas is not None:
        cdr_bandicoot = cdr_bandicoot.join(antennas.select(['antenna_id', 'latitude', 'longitude']),
                                           on='antenna_id', how='left')
    
    cdr_bandicoot = cdr_bandicoot.na.fill('')
    
    return cdr_bandicoot


def long_join_pandas(dfs: List[PandasDataFrame], on: str,
                     how: Union[Literal['left'], Literal['right'],
                                Literal['outer'], Literal['inner']]) -> PandasDataFrame:
    """
    Join list of pandas dfs

    Args:
        dfs: list of pandas df
        on: column on which to join
        how: type of join

    Returns: single joined pandas df
    """
    df = dfs[0]
    for i in range(1, len(dfs)):
        df = df.merge(dfs[i], on=on, how=how)
    return df


def long_join_pyspark(dfs: List[SparkDataFrame], on: str, how: str) -> SparkDataFrame:
    """
    Join list of spark dfs

    Args:
        dfs: list of spark df
        on: column on which to join
        how: type of join

    Returns: single joined spark df
    """
    df = dfs[0]
    for i in range(1, len(dfs)):
        df = df.join(dfs[i], on=on, how=how)
    return df


def strictly_increasing(L: List[float]) -> bool:
    # Check that the list's values are strictly increasing
    return all(x < y for x, y in zip(L, L[1:]))


def check_columns_exist(data: Union[PandasDataFrame, SparkDataFrame],
                        columns: List[str],
                        data_name: str = '') -> None:
    """
    Check that list of columns is present in df

    Args:
        data: df
        columns: columns to check
        data_name: name of dataset to print out
    """
    for c in columns:
        if c not in data.columns:
            raise ValueError('Column ' + c + ' not in data ' + data_name + '.')


def check_column_types(data: PandasDataFrame, continuous: List[str], categorical: List[str], binary: List[str]) -> None:
    """
    Try to identify issues with column types and values

    Args:
        data: pandas df
        continuous: continuous columns to check
        categorical: categorical columns to check
        binary: binary columns to check
    """
    for c in continuous:
        n_unique = len(data[c].unique())
        if n_unique < 20:
            print('Warning: Column ' + c + ' is of continuous type but has fewer than 20 (%i) unique values.' % n_unique)
    for c in categorical:
        n_unique = len(data[c].unique())
        if n_unique > 20:
            print('Warning: Column ' + c + ' is of categorical type but has more than 20 (%i) unique values.' % n_unique)
    for c in binary:
        if set(data[c].dropna().astype('int')) != {0, 1}:
            raise ValueError('Column ' + c + ' is labeled as binary but does not contain only 0 and 1.')


# Source: https://stackoverflow.com/questions/38641691/weighted-correlation-coefficient-with-pandas
def weighted_mean(x: ndarray, w: ndarray) -> float:
    return np.sum(x * w) / np.sum(w)


def weighted_cov(x: ndarray, y: ndarray, w: ndarray) -> float:
    return np.sum(w * (x - weighted_mean(x, w)) * (y - weighted_mean(y, w))) / np.sum(w)


def weighted_corr(x: ndarray, y: ndarray, w: ndarray) -> float:
    return weighted_cov(x, y, w) / np.sqrt(weighted_cov(x, x, w) * weighted_cov(y, y, w))


def build_config_from_file(config_file_path_string: str) -> Box:
    """
    Build the config Box (dictionary) from file and convert file paths to pathlib.Path objects, taking into 
    account that some paths are expected to be defined relative to other paths, returning a Box containing
    file paths.
    """
    
    def recursively_convert_to_path_and_resolve(dict_or_path_string, path_root):
        
        if dict_or_path_string is None:
            return None
        
        elif isinstance(dict_or_path_string, str):
            
            path = Path(dict_or_path_string)
            
            if os.path.isabs(path):
                return path
            
            return path_root / path
        
        else:
            
            new_dict = {}
            for key, value in dict_or_path_string.items():
                
                new_dict[key] = recursively_convert_to_path_and_resolve(value, path_root)

            return new_dict
                
    with open(config_file_path_string, 'r') as config_file:
        config_dict = yaml_load(config_file, Loader=FullLoader)

    input_path_dict = config_dict['path']
    
    processed_path_dict = {}
    
    # get the working directory path
    working_directory_path = Path(input_path_dict['working']['directory_path'])
    if not os.path.isabs(working_directory_path):
        # raise ValueError(f'expected absolute path to working directory; got {working_directory_path} instead.')
        # This is only allowed because our tests rely on it. TODO: Change tests so this isn't necessary
        working_directory_path = Path(__file__).parent.parent / working_directory_path
    
    # get the top level input data directory
    input_data_directory_path = Path(input_path_dict['input_data']['directory_path'])
    if not os.path.isabs(input_data_directory_path):
        # raise ValueError(f'expected absolute path to input data directory; got {input_data_directory_path} instead.')
        # This is only allowed because our tests rely on it. TODO: Change tests so this isn't necessary
        input_data_directory_path = Path(__file__).parent.parent / input_data_directory_path

    # now recursively turn the rest of the path strings into Path objects
    processed_path_dict['input_data'] = recursively_convert_to_path_and_resolve(input_path_dict['input_data'], input_data_directory_path)
    processed_path_dict['working'] = recursively_convert_to_path_and_resolve(input_path_dict['working'], working_directory_path)

    # Correct the top-level directorypaths which should be interpreted differently: Otherwise the final leg of the directory
    # path will be repeated.
    processed_path_dict['input_data']['directory_path'] = input_data_directory_path
    processed_path_dict['working']['directory_path'] = working_directory_path

    config_dict['path'] = processed_path_dict
    
    return Box(config_dict)
