from box import Box  # type: ignore[import]
import numpy as np
from numpy import ndarray
import os
import pandas as pd
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.functions import col, date_format, lit
from pyspark.sql import SparkSession
import shutil
from typing import List, Tuple, Union
from typing_extensions import Literal


def get_spark_session(cfg: Box) -> SparkSession:
    """
    Gets or creates spark session, with context and logging preferences set
    """
    # Build spark session
    spark = SparkSession \
        .builder \
        .appName(cfg.spark.app_name) \
        .config("spark.sql.files.maxPartitionBytes", cfg.spark.files.max_partition_bytes) \
        .config("spark.driver.memory", cfg.spark.driver.memory) \
        .config("spark.driver.maxResultSize", cfg.spark.driver.max_result_size)\
        .getOrCreate()
    spark.sparkContext.setLogLevel(cfg.spark.loglevel)
    return spark


def save_df(df: SparkDataFrame, outfname: str, sep: str = ',') -> None:
    """
    Saves spark dataframe to csv file, using work-around to deal with spark's automatic partitioning and naming
    """
    outfolder = outfname[:-4]
    df.repartition(1).write.csv(path=outfolder, mode="overwrite", header="true", sep=sep)
    # Work around to deal with spark automatic naming
    old_fname = [fname for fname in os.listdir(outfolder) if fname[-4:] == '.csv'][0]
    os.rename(outfolder + '/' + old_fname, outfname)
    shutil.rmtree(outfolder)


def save_parquet(df: SparkDataFrame, outfname: str) -> None:
    """
    Save spark dataframe to parquet file
    """
    df.write.mode('overwrite').parquet(outfname)


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


def make_dir(fname: str, remove: bool = False) -> None:
    """
    Create new directory

    Args:
        fname: directory path
        remove: whether to remove directory if already present
    """
    if os.path.isdir(fname) and remove:
        shutil.rmtree(fname)
    os.makedirs(fname, exist_ok=True)


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