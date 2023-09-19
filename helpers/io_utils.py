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

from pathlib import Path
from typing import Dict, List, Optional, Union
import re

import geopandas as gpd  # type: ignore[import]
from box import Box
from geopandas import GeoDataFrame
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, date_trunc, lit, to_timestamp
from pyspark.sql.types import StringType


from helpers.utils import get_spark_session


def load_generic(cfg: Box,
                 fpath: Optional[Path] = None,
                 df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None) -> SparkDataFrame:
    """
    Args:
        cfg: box object containing config data
        file_path: path to file or folder with files
        df: pandas or spark df, if already loaded

    Returns: loaded spark df
    """
    spark = get_spark_session(cfg)

    # Load from file
    if fpath is not None:
        # Load data if in a single file
        if fpath.is_file():
            df = spark.read.csv(str(fpath), header=True)

        # Load data if in chunks
        else:
            df = spark.read.csv(str(fpath / '*.csv'), header=True)

    # Load from pandas dataframe
    elif df is not None:
        if not isinstance(df, SparkDataFrame):
            df = spark.createDataFrame(df)

    # Issue with filename/dataframe provided
    else:
        raise ValueError('No filename or pandas/spark dataframe provided.')

    return df


def check_cols(df: Union[GeoDataFrame, PandasDataFrame, SparkDataFrame],
               required_cols: List[str],
               error_msg: str) -> None:
    """
    Check that the df has all required columns

    Args:
        df: spark df
        required_cols: columns that are required
        error_msg: error message to print if some columns are missing
    """
    if not set(required_cols).intersection(set(df.columns)) == set(required_cols):
        raise ValueError(error_msg)


def check_colvalues(df: SparkDataFrame, colname: str, colvalues: list, error_msg: str) -> None:
    """
    Check that a column has all required values

    Args:
        df: spark df
        colname: column to check
        colvalues: requires values
        error_msg: error message to print if values don't match
    """
    if set(df.select(colname).distinct().rdd.map(lambda r: r[0]).collect()).union(set(colvalues)) != set(colvalues):
        raise ValueError(error_msg)


def standardize_col_names(df: SparkDataFrame, col_names: Dict[str, str]) -> SparkDataFrame:
    """
    Rename columns, as specified in config file, to standard format

    Args:
        df: spark df
        col_names: mapping between standard column names and existing ones

    Returns: spark df with standardized column names

    """
    col_mapping = {v: k for k, v in col_names.items()}

    for col in df.columns:
        if col in col_mapping:
            df = df.withColumnRenamed(col, col_mapping[col])

    return df


def load_cdr(cfg: Box,
             fpath: Optional[Path] = None,
             df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None,
             verify: bool = True) -> SparkDataFrame:
    """
    Load CDR data into spark df

    Args:
        cfg: box object containing config data
        fpath: path to file or folder with files
        df: pandas or spark df, if already loaded
        verify: whether to check if right columns and values are present

    Returns: spark df
    """
    spark = get_spark_session(cfg)
    # load data as generic df and standardize column_names
    if df is not None:
        if isinstance(df, PandasDataFrame):
            cdr = spark.createDataFrame(df)
        elif isinstance(df, SparkDataFrame):
            cdr = df
        else:
            raise TypeError("The dataframe provided should be a spark or pandas df.")
    elif fpath is not None:
        cdr = load_generic(cfg, fpath=fpath, df=df)
    else:
        raise ValueError('No filename or pandas/spark dataframe provided.')
    cdr = standardize_col_names(cdr, cfg.col_names.cdr)

    if verify:
        # Check that required columns are present
        required_cols = ['txn_type', 'caller_id', 'recipient_id', 'timestamp', 'duration', 'international']
        error_msg = (
            f"CDR data format incorrect. CDR must include the following columns: {', '.join(required_cols)}, "
            f"instead found {', '.join(cdr.columns)}"
        )
        check_cols(cdr, required_cols, error_msg)

        # Check txn_type column
        error_msg = 'CDR format incorrect. Column txn_type can only include call and text.'
        check_colvalues(cdr, 'txn_type', ['call', 'text'], error_msg)

        # Clean international column
        error_msg = 'CDR format incorrect. Column international can only include domestic, international, and other.'
        check_colvalues(cdr, 'international', ['domestic', 'international', 'other'], error_msg)
        
        # if no recipient antennas are present, add a null column to enable the featurizer to work
        # TODO(leo): Consider cleaning up featurizer logic so this isn't needed.
        if 'recipient_antenna' not in cdr.columns:
            cdr = cdr.withColumn('recipient_antenna', lit(None).cast(StringType()))

    # Clean timestamp column
    cdr = clean_timestamp_and_add_day_column(cdr, 'timestamp')

    # Clean duration column
    cdr = cdr.withColumn('duration', col('duration').cast('float'))

    return cdr

def load_labels(
    cfg: Box,
    fpath: Path = None,
    verify: bool = True
) -> SparkDataFrame:

        """
        Load labels on which to train ML model.
        """
        
        spark = get_spark_session(cfg)
        
        # If the user specified column names for labels file, rename columns accordingly
        labels = load_generic(cfg, fpath=fpath)
        if 'labels' in cfg.col_names:
            labels = standardize_col_names(labels, cfg.col_names.labels)
        if verify:
            required_cols = ['name', 'label']
            error_msg = (
                f"Labels data format incorrect. Labels must include the following columns: {', '.join(required_cols)}, "
                f"instead found {', '.join(labels.columns)}"
            )
            check_cols(labels, required_cols, error_msg)

        if 'weight' not in labels.columns:
            labels = labels.withColumn('weight', lit(1))

        return labels.select(['name', 'label', 'weight'])

                             
def load_antennas(cfg: Box,
                  fpath: Optional[Path] = None,
                  df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None,
                  verify: bool = True) -> SparkDataFrame:
    """
    Load antennas' dataset, and print % of antennas that are missing coordinates

    Args:
        cfg: box object containing config data
        fpath: path to file
        df: pandas or spark df, if already loaded
        verify: whether to check if right columns and values are present

    Returns: spark df
    """
    spark = get_spark_session(cfg)
    # load data as generic df and standardize column_names
    if df is not None:
        if isinstance(df, PandasDataFrame):
            antennas = spark.createDataFrame(df)
        elif isinstance(df, SparkDataFrame):
            antennas = df
        else:
            raise TypeError("The dataframe provided should be a spark or pandas df.")
    elif fpath is not None:
        antennas = load_generic(cfg, fpath=fpath, df=df)
    else:
        raise ValueError('No filename or pandas/spark dataframe provided.')
    antennas = standardize_col_names(antennas, cfg.col_names.antennas)
    if verify:
        required_cols = ['antenna_id', 'latitude', 'longitude']
        error_msg = (
            f"Antenna data format incorrect. Antenna data must include the following columns: {', '.join(required_cols)}, "
            f"instead found {', '.join(antennas.columns)}"
        )
        check_cols(antennas, required_cols, error_msg)

        antennas = antennas.withColumn('latitude', col('latitude').cast('float')).withColumn('longitude',
                                                                                             col('longitude').cast(
                                                                                                 'float'))
        print('Warning: %i antennas missing location' % (
                antennas.count() - antennas.select(['latitude', 'longitude']).na.drop().count()))

    return antennas


def load_recharges(cfg: Box,
                   fpath: Optional[Path] = None,
                   df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None) -> SparkDataFrame:
    """
    Load recharges' dataset

    Args:
        cfg: box object containing config data
        fpath: path to file or folder with files
        df: pandas or spark df, if already loaded

    Returns: spark df
    """
    spark = get_spark_session(cfg)
    # load data as generic df and standardize column_names
    if df is not None:
        if isinstance(df, PandasDataFrame):
            recharges = spark.createDataFrame(df)
        elif isinstance(df, SparkDataFrame):
            recharges = df
        else:
            raise TypeError("The dataframe provided should be a spark or pandas df.")
    elif fpath is not None:
        recharges = load_generic(cfg, fpath=fpath, df=df)
    else:
        raise ValueError('No filename or pandas/spark dataframe provided.')
    recharges = standardize_col_names(recharges, cfg.col_names.recharges)

    required_cols = ['caller_id', 'amount', 'timestamp']
    error_msg = (
        f"Recharges data format incorrect. Recharges must include the following columns: {', '.join(required_cols)}, "
        f"instead found {', '.join(recharges.columns)}"
    )
    check_cols(recharges, required_cols, error_msg)
    # Clean timestamp column
    recharges = clean_timestamp_and_add_day_column(recharges, 'timestamp')

    # Clean amount column
    recharges = recharges.withColumn('amount', col('amount').cast('float'))

    return recharges


def load_mobiledata(cfg: Box,
                    fpath: Optional[Path] = None,
                    df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None) -> SparkDataFrame:
    """
    Load mobile data dataset

    Args:
        cfg: box object containing config data
        fpath: path to file or folder with files
        df: pandas or spark df, if already loaded

    Returns: spark df
    """
    spark = get_spark_session(cfg)
    # load data as generic df and standardize column_names
    if df is not None:
        if isinstance(df, PandasDataFrame):
            mobiledata = spark.createDataFrame(df)
        elif isinstance(df, SparkDataFrame):
            mobiledata = df
        else:
            raise TypeError("The dataframe provided should be a spark or pandas df.")
    elif fpath is not None:
        mobiledata = load_generic(cfg, fpath=fpath, df=df)
    else:
        raise ValueError('No filename or pandas/spark dataframe provided.')

    mobiledata = standardize_col_names(mobiledata, cfg.col_names.mobiledata)

    required_cols = ['caller_id', 'volume', 'timestamp']
    error_msg = (
        f"Mobile data format incorrect. Mobile data records must include the following columns: {', '.join(required_cols)}, "
        f"instead found {', '.join(mobiledata.columns)}"
    )
    check_cols(mobiledata, required_cols, error_msg)

    # Clean timestamp column
    mobiledata = clean_timestamp_and_add_day_column(mobiledata, 'timestamp')

    # Clean duration column
    mobiledata = mobiledata.withColumn('volume', col('volume').cast('float'))

    return mobiledata


def load_mobilemoney(cfg: Box,
                     fpath: Optional[Path] = None,
                     df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None,
                     verify: bool = True) -> SparkDataFrame:
    """
    Load mobile money dataset

    Args:
        cfg: box object containing config data
        fpath: path to file or folder with files
        df: pandas or spark df, if already loaded
        verify: whether to check if right columns and values are present

    Returns: spark df
    """
    spark = get_spark_session(cfg)
    # load data as generic df and standardize column_names
    if df is not None:
        if isinstance(df, PandasDataFrame):
            mobilemoney = spark.createDataFrame(df)
        elif isinstance(df, SparkDataFrame):
            mobilemoney = df
        else:
            raise TypeError("The dataframe provided should be a spark or pandas df.")
    elif fpath is not None:
        mobilemoney = load_generic(cfg, fpath=fpath, df=df)
    else:
        raise ValueError('No filename or pandas/spark dataframe provided.')
    mobilemoney = standardize_col_names(mobilemoney, cfg.col_names.mobilemoney)

    if verify:
        # Check that required columns are present
        required_cols = ['txn_type', 'caller_id', 'recipient_id', 'timestamp', 'amount']
        error_msg = (
            f"Mobile money data format incorrect. Mobile money must include the following columns: {', '.join(required_cols)}, "
            f"instead found {', '.join(mobilemoney.columns)}"
        )

        check_cols(mobilemoney, required_cols, error_msg)

        # Check txn_type column
        txn_types = ['cashin', 'cashout', 'p2p', 'billpay', 'other']
        error_msg = 'Mobile money format incorrect. Column txn_type can only include ' + ', '.join(txn_types)
        check_colvalues(mobilemoney, 'txn_type', txn_types, error_msg)

    # Clean timestamp column
    mobilemoney = clean_timestamp_and_add_day_column(mobilemoney, 'timestamp')

    # Clean duration column
    mobilemoney = mobilemoney.withColumn('amount', col('amount').cast('float'))

    # Clean balance columns
    for c in mobilemoney.columns:
        if 'balance' in c:
            mobilemoney = mobilemoney.withColumn(c, col(c).cast('float'))

    return mobilemoney


def load_shapefile(fpath: Path) -> GeoDataFrame:
    """
    Load shapefile and make sure it has the right columns

    Args:
        fpath: path to file, which can be .shp or .geojson

    Returns: GeoDataFrame

    """
    shapefile = gpd.read_file(fpath)

    # Verify that columns are correct
    required_cols = ['region', 'geometry']
    error_msg = (
        f"Shapefile data format incorrect. Shapefile must include the following columns: {', '.join(required_cols)}, "
        f"instead found {', '.join(shapefile.columns)}"
    )
    check_cols(shapefile, required_cols, error_msg)

    # Verify that the geometry column has been loaded correctly
    assert shapefile.dtypes['geometry'] == 'geometry'

    shapefile['region'] = shapefile['region'].astype(str)

    return shapefile


def clean_timestamp_and_add_day_column(
    df: SparkDataFrame,
    existing_timestamp_column_name: str
):

    # Check the first row for time info, and assume the format is consistent
    existing_timestamp_sample = df.take(1)[0][existing_timestamp_column_name]
    timestamp_with_time_regex = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"

    has_time_info = bool(re.match(timestamp_with_time_regex, existing_timestamp_sample))

    timestamp_format = (
        'yyyy-MM-dd HH:mm:ss' if has_time_info else 'yyyy-MM-dd'
    )

    return (
        df
        .withColumn(
            'timestamp',
            to_timestamp(existing_timestamp_column_name, timestamp_format)
        )
        .withColumn('day', date_trunc('day', col('timestamp')))
    )


def load_phone_numbers_to_featurize(
    cfg: Box,
    fpath: Optional[Path] = None,
    df: Optional[Union[SparkDataFrame, PandasDataFrame]] = None,
    verify: bool = True
) -> SparkDataFrame:

    phone_numbers_to_featurize = load_generic(cfg, fpath=fpath, df=df)

    phone_numbers_to_featurize = standardize_col_names(phone_numbers_to_featurize, cfg.col_names.phone_numbers_to_featurize)

    if verify:
        # Check that required columns are present
        required_cols = ['phone_number']
        error_msg = (
            f"Phone numbers to featurize data format incorrect. Must include the following columns: {', '.join(required_cols)}, "
            f"instead found {', '.join(phone_numbers_to_featurize.columns)}"
        )

        check_cols(phone_numbers_to_featurize, required_cols, error_msg)

    return phone_numbers_to_featurize.select('phone_number')
