# Copyright ©2022-2023. The Regents of the University of California (Regents). 
# All Rights Reserved. 

# Permission to use, copy, modify, and distribute this software and its 
# documentation for educational, research, and not-for-profit purposes, without
# fee and without a signed licensing agreement, is hereby granted, provided that 
# the above copyright notice, this paragraph and the following two paragraphs 
# appear in all copies, modifications, and distributions. Contact The Office of
# Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, 
# CA 94720-1620, (510) 643-7201, otl@berkeley.edu, 
# http://ipira.berkeley.edu/industry-info for commercial licensing 
# opportunities.

# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, 
# SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING
# OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS 
# BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED 
# HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE 
# MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd  # type: ignore[import]
from box import Box
from geopandas import GeoDataFrame
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, date_trunc, to_timestamp

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
            df = spark.read.csv(str(fpath + '/*.csv'), header=True)

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
        error_msg = 'CDR format incorrect. CDR must include the following columns: ' + ', '.join(required_cols)
        check_cols(cdr, required_cols, error_msg)

        # Check txn_type column
        error_msg = 'CDR format incorrect. Column txn_type can only include call and text.'
        check_colvalues(cdr, 'txn_type', ['call', 'text'], error_msg)

        # Clean international column
        error_msg = 'CDR format incorrect. Column international can only include domestic, international, and other.'
        check_colvalues(cdr, 'international', ['domestic', 'international', 'other'], error_msg)

    # Clean timestamp column
    cdr = cdr.withColumn('timestamp', to_timestamp(cdr['timestamp'], 'yyyy-MM-dd HH:mm:ss')) \
        .withColumn('day', date_trunc('day', col('timestamp')))

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
            error_msg = f'Labels must include columns {required_cols}.'
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
        error_msg = 'Antenna format incorrect. Antenna dataset must include the following columns: ' + ', '.join(
            required_cols)
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

    # Clean timestamp column
    recharges = recharges.withColumn('timestamp', to_timestamp('timestamp', 'yyyy-MM-dd HH:mm:ss')) \
        .withColumn('day', date_trunc('day', col('timestamp')))

    # Clean duration column
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

    # Clean timestamp column
    mobiledata = mobiledata.withColumn('timestamp', to_timestamp('timestamp', 'yyyy-MM-dd HH:mm:ss')) \
        .withColumn('day', date_trunc('day', col('timestamp')))

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
        error_msg = 'Mobile money format incorrect. Mobile money records must include the following columns: ' + \
                    ', '.join(required_cols)
        check_cols(mobilemoney, required_cols, error_msg)

        # Check txn_type column
        txn_types = ['cashin', 'cashout', 'p2p', 'billpay', 'other']
        error_msg = 'Mobile money format incorrect. Column txn_type can only include ' + ', '.join(txn_types)
        check_colvalues(mobilemoney, 'txn_type', txn_types, error_msg)

    # Clean timestamp column
    mobilemoney = mobilemoney.withColumn('timestamp', to_timestamp(mobilemoney['timestamp'], 'yyyy-MM-dd HH:mm:ss')) \
        .withColumn('day', date_trunc('day', col('timestamp')))

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
    error_msg = 'Shapefile format incorrect. Shapefile must include the following columns: ' + ', '.join(required_cols)
    check_cols(shapefile, required_cols, error_msg)

    # Verify that the geometry column has been loaded correctly
    assert shapefile.dtypes['geometry'] == 'geometry'

    shapefile['region'] = shapefile['region'].astype(str)

    return shapefile

