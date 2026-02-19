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

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Union, Optional
import warnings

from importlib_resources import files as importlib_resources_files
import numpy as np
import pandas as pd
import pyarrow as pa
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from box import Box
from numpy import ndarray
from pandas import DataFrame as PandasDataFrame
from pandas.api.types import is_numeric_dtype
from typing_extensions import Literal
from yaml import FullLoader, load as yaml_load

# Type alias for Dask DataFrames
DaskDataFrame = dd.DataFrame


def get_dask_client(cfg: Box) -> Client:
    """
    Gets or creates Dask client with configuration preferences set
    
    Args:
        cfg: Configuration box containing dask settings
        
    Returns:
        Configured Dask client
    """
    # Check if client already exists
    try:
        client = Client.current()
        return client
    except ValueError:
        # No client exists, create one
        pass
    
    # Extract dask configuration
    dask_cfg = cfg.get('dask', Box())
    
    # Build cluster configuration
    n_workers = dask_cfg.get('n_workers', 4)
    threads_per_worker = dask_cfg.get('threads_per_worker', 1)
    memory_limit = dask_cfg.get('memory_limit', 'auto')
    silence_logs = dask_cfg.get('silence_logs', True)
    
    if silence_logs:
        # Reduce log noise from distributed/Dask so featurizer progress prints are visible
        for name in ('distributed', 'dask', 'dask.distributed'):
            logging.getLogger(name).setLevel(logging.ERROR)
    
    # Create local cluster
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        silence_logs=silence_logs,
    )
    
    # Create and connect client
    client = Client(cluster)
    
    return client


def save_df(df: Union[DaskDataFrame, PandasDataFrame], out_file_path: Path, sep: str = ',', single_file=True) -> None:
    """
    Saves dask/pandas dataframe to csv file
    
    Args:
        df: Dask or Pandas DataFrame
        out_file_path: Path to output file
        sep: Separator character
        single_file: If True, write to a single file. If False, write partitioned files.
    """
    if single_file:
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(df, PandasDataFrame):  # Pandas case
            df.to_csv(str(out_file_path), header=True, sep=sep, index=False)
        elif isinstance(df, dd.DataFrame):  # Dask case
            # Convert to pandas for single file output
            pandas_df = df.compute()
            pandas_df.to_csv(str(out_file_path), header=True, sep=sep, index=False)
        else:
            raise TypeError("Not a dask or pandas dataframe")
    else:
        if isinstance(df, PandasDataFrame):  # Pandas case
            out_file_path.mkdir(parents=False, exist_ok=True)
            df.to_csv(str(out_file_path / "0.csv"), header=True, sep=sep, index=False)
        elif isinstance(df, dd.DataFrame):  # Dask case
            # Write partitioned output
            out_file_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(str(out_file_path / "*.csv"), header=True, sep=sep, index=False)
        else:
            raise TypeError("Not a dask or pandas dataframe")


def read_csv(client: Optional[Client], file_path: Path, **kwargs) -> DaskDataFrame:
    """
    A wrapper around dd.read_csv which accepts pathlib.Path objects as input.
    
    Args:
        client: Dask client (kept for API compatibility, not used)
        file_path: Path to CSV file(s)
        **kwargs: Additional arguments to pass to dd.read_csv
        
    Returns:
        Dask DataFrame
    """
    # Dask forwards to pandas.read_csv; normalize common Spark-style kwargs.
    header = kwargs.get("header", None)
    if isinstance(header, bool):
        # Spark uses header=True/False; pandas expects None/int/list-like.
        if header:
            kwargs.pop("header", None)  # default is header='infer'
        else:
            kwargs["header"] = None

    # Avoid dtype mismatch across partitions when integer columns contain missing values.
    # If callers want strict dtypes they must pass dtype= explicitly.
    kwargs.setdefault("assume_missing", True)

    # Set default dtype for phone number columns to string
    dtype = kwargs.pop('dtype', None)
    if dtype is None:
        dtype = {'caller_id': 'str', 'recipient_id': 'str', 'name': 'str'}
    
    return dd.read_csv(str(file_path), dtype=dtype, **kwargs)


def read_parquet(client: Optional[Client], file_path: Path, **kwargs) -> DaskDataFrame:
    """
    A wrapper around dd.read_parquet which accepts pathlib.Path objects as input.
    
    Args:
        client: Dask client (kept for API compatibility, not used)
        file_path: Path to Parquet file(s)
        **kwargs: Additional arguments to pass to dd.read_parquet
        
    Returns:
        Dask DataFrame
    """
    return dd.read_parquet(str(file_path), **kwargs)


def _pandas_dtype_to_arrow(dtype) -> pa.DataType:
    """Map pandas dtype to PyArrow type for consistent parquet schema across partitions."""
    name = str(dtype).lower()
    if name in ('object', 'string', 'str'):
        return pa.string()
    if name in ('int8', 'int16', 'int32', 'int64', 'int', 'int_'):
        return pa.int64()
    if name in ('uint8', 'uint16', 'uint32', 'uint64'):
        return pa.uint64()
    if 'int64' in name or 'int32' in name or 'int16' in name or 'int8' in name:
        return pa.int64()
    if name in ('float16', 'float32', 'float64', 'float', 'float_'):
        return pa.float64()
    if 'float' in name:
        return pa.float64()
    if name in ('bool', 'boolean'):
        return pa.bool_()
    if name.startswith('datetime64') or 'datetime' in name:
        return pa.timestamp('us')
    if name.startswith('timedelta64') or 'timedelta' in name:
        return pa.duration('us')
    # default for unknown (e.g. categorical, period): string
    return pa.string()


def save_parquet(df: Union[DaskDataFrame, PandasDataFrame], out_directory_path: Path) -> None:
    """
    Save dask or pandas dataframe to parquet file(s).
    Uses an explicit PyArrow schema for Dask writes so all partitions share the same
    schema, avoiding "Repetition level histogram size mismatch" when reading.
    """
    def _object_cols(dtypes):
        return [c for c in dtypes.index if dtypes[c] == object or str(dtypes[c]) == 'object']

    if isinstance(df, dd.DataFrame):
        out_directory_path.mkdir(parents=True, exist_ok=True)
        obj_cols = _object_cols(df.dtypes)
        if obj_cols:
            df = df.assign(**{c: df[c].astype(str) for c in obj_cols})
        # Explicit schema so every partition writes the same schema (avoids read errors)
        schema = pa.schema([
            (c, _pandas_dtype_to_arrow(df.dtypes[c])) for c in df.columns
        ])
        df.to_parquet(
            str(out_directory_path),
            write_index=False,
            overwrite=True,
            schema=schema,
        )

    elif isinstance(df, PandasDataFrame):
        out_directory_path.mkdir(parents=True, exist_ok=True)
        obj_cols = _object_cols(df.dtypes)
        if obj_cols:
            df = df.copy()
            for c in obj_cols:
                df[c] = df[c].astype(str)
        df.to_parquet(out_directory_path / '0.parquet', index=False)


def filter_dates_dataframe(df: DaskDataFrame,
                           start_date: str, end_date: str, colname: str = 'timestamp') -> DaskDataFrame:
    """
    Filter dataframe rows whose timestamp is outside [start_date, end_date]

    Args:
        df: dask dataframe
        start_date: initial date to keep
        end_date: last date to include (inclusive)
        colname: name of timestamp column

    Returns: 
        Filtered dask dataframe
    """
    if colname not in df.columns:
        raise ValueError('Cannot filter dates because missing timestamp column')
    
    # Always convert to datetime so we never compare string to Timestamp (e.g. when
    # Parquet has large_string or mixed partitions).
    df = df.assign(**{colname: dd.to_datetime(df[colname], errors="coerce")})
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(value=1, unit='days')
    df = df[(df[colname] >= start_dt) & (df[colname] < end_dt)]
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
    """Flatten a list of lists"""
    return [item for sublist in lst for item in sublist]


def cdr_bandicoot_format(cdr: DaskDataFrame, antennas: DaskDataFrame, cfg: Box) -> DaskDataFrame:
    """
    Convert CDR df into format that can be used by bandicoot

    Args:
        cdr: dask df with CDRs
        antennas: antenna dataframe
        cfg: box object with cdr column names

    Returns: 
        Dask df in bandicoot format
    """
    cols = list(cfg.keys())

    # Create outgoing transactions
    outgoing = cdr[cols].copy()
    outgoing = outgoing.rename(columns={
        'txn_type': 'interaction',
        'caller_id': 'name',
        'recipient_id': 'correspondent_id',
        'timestamp': 'datetime',
        'duration': 'call_duration',
        'caller_antenna': 'antenna_id'
    })
    outgoing['direction'] = 'out'
    outgoing = outgoing.drop(columns=['recipient_antenna'], errors='ignore')

    # Create incoming transactions
    incoming = cdr[cols].copy()
    incoming = incoming.rename(columns={
        'txn_type': 'interaction',
        'recipient_id': 'name',
        'caller_id': 'correspondent_id',
        'timestamp': 'datetime',
        'duration': 'call_duration',
        'recipient_antenna': 'antenna_id'
    })
    incoming['direction'] = 'in'
    incoming = incoming.drop(columns=['caller_antenna'], errors='ignore')

    # Combine
    cdr_bandicoot = dd.concat([outgoing, incoming], axis=0)
    
    # Convert call_duration to string
    cdr_bandicoot['call_duration'] = cdr_bandicoot['call_duration'].astype('Int64').astype(str)
    
    # Format datetime
    cdr_bandicoot['datetime'] = dd.to_datetime(cdr_bandicoot['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Join with antennas if available
    if antennas is not None:
        # Ensure same type for merge (CDR may have float antenna ids, antennas may have str)
        cdr_bandicoot = cdr_bandicoot.assign(antenna_id=cdr_bandicoot['antenna_id'].astype(str))
        antenna_cols = antennas[['antenna_id', 'latitude', 'longitude']].assign(
            antenna_id=antennas['antenna_id'].astype(str)
        )
        cdr_bandicoot = cdr_bandicoot.merge(antenna_cols, on='antenna_id', how='left')
    
    # Fill missing values
    cdr_bandicoot = cdr_bandicoot.fillna('')
    
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

    Returns: 
        Single joined pandas df
    """
    if len(dfs) == 0:
        return None
    df = dfs[0]
    for i in range(1, len(dfs)):
        df = df.merge(dfs[i], on=on, how=how)
    return df


def long_join_dask(dfs: List[DaskDataFrame], on: str, how: str) -> DaskDataFrame:
    """
    Join list of dask dfs

    Args:
        dfs: list of dask df
        on: column on which to join
        how: type of join

    Returns:
        Single joined dask df
    """
    if len(dfs) == 0:
        return None
    df = dfs[0]
    if on not in df.columns:
        raise KeyError(
            f"Join key '{on}' not found in first dataframe. "
            f"Columns: {list(df.columns)}"
        )
    for i in range(1, len(dfs)):
        if on not in dfs[i].columns:
            raise KeyError(
                f"Join key '{on}' not found in dataframe index {i}. "
                f"Columns: {list(dfs[i].columns)}"
            )
        # Cast join column to same type (string) to avoid merge on string vs int64
        df = df.assign(**{on: df[on].astype(str)})
        other = dfs[i].assign(**{on: dfs[i][on].astype(str)})
        df = df.merge(other, on=on, how=how)
    return df


# Backwards compatibility alias
long_join_pyspark = long_join_dask


def strictly_increasing(L: List[float]) -> bool:
    """Check that the list's values are strictly increasing"""
    return all(x < y for x, y in zip(L, L[1:]))


def check_columns_exist(data: Union[PandasDataFrame, DaskDataFrame],
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
    """Calculate weighted mean"""
    return np.sum(x * w) / np.sum(w)


def weighted_cov(x: ndarray, y: ndarray, w: ndarray) -> float:
    """Calculate weighted covariance"""
    return np.sum(w * (x - weighted_mean(x, w)) * (y - weighted_mean(y, w))) / np.sum(w)


def weighted_corr(x: ndarray, y: ndarray, w: ndarray) -> float:
    """Calculate weighted correlation"""
    return weighted_cov(x, y, w) / np.sqrt(weighted_cov(x, x, w) * weighted_cov(y, y, w))


def get_data_format():
    """Load data format configuration"""
    data_format_path = importlib_resources_files('data_format') / 'data_format.yml'

    with open(data_format_path, 'r') as data_format_file:
        data_format_dict = yaml_load(data_format_file, Loader=FullLoader)

    return Box(data_format_dict)


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
        # This is only allowed because our tests rely on it. TODO: Change tests so this isn't necessary
        working_directory_path = Path(__file__).parent.parent / working_directory_path
    
    # get the top level input data directory
    input_data_directory_path = Path(input_path_dict['input_data']['directory_path'])
    if not os.path.isabs(input_data_directory_path):
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


def filter_by_phone_numbers_to_featurize(
    phone_numbers_to_featurize: Optional[DaskDataFrame],
    df: Union[DaskDataFrame, PandasDataFrame],
    phone_number_column_name: str
) -> Union[DaskDataFrame, PandasDataFrame]:
    """
    Filter dataframe to only include specified phone numbers
    
    Args:
        phone_numbers_to_featurize: DataFrame with 'phone_number' column
        df: DataFrame to filter
        phone_number_column_name: Name of column in df containing phone numbers
        
    Returns:
        Filtered dataframe
    """
    if phone_numbers_to_featurize is None:
        return df

    # Cast join columns to str so dtypes match (avoids str vs string merge warning and object columns)
    if isinstance(df, dd.DataFrame):
        df = df.assign(**{phone_number_column_name: df[phone_number_column_name].astype(str)})
        phone_numbers_to_featurize = phone_numbers_to_featurize.assign(
            phone_number=phone_numbers_to_featurize['phone_number'].astype(str)
        )
        return df.merge(
            phone_numbers_to_featurize,
            left_on=phone_number_column_name,
            right_on='phone_number',
            how='inner'
        ).drop(columns='phone_number')

    else:
        # Pandas case
        phone_numbers_to_featurize_pandas = phone_numbers_to_featurize.compute() if isinstance(phone_numbers_to_featurize, dd.DataFrame) else phone_numbers_to_featurize
        df = df.copy()
        df[phone_number_column_name] = df[phone_number_column_name].astype(str)
        phone_numbers_to_featurize_pandas = phone_numbers_to_featurize_pandas.copy()
        phone_numbers_to_featurize_pandas['phone_number'] = phone_numbers_to_featurize_pandas['phone_number'].astype(str)
        return df.merge(
            phone_numbers_to_featurize_pandas,
            left_on=phone_number_column_name,
            right_on='phone_number',
            how='inner'
        ).drop(columns='phone_number')


# For testing only. Compare two dataframes that are expected to be similar or identical. Obtain info
# about matches/mismatches row- and column-wise.
def testonly_compare_dataframes(left: pd.DataFrame, right: pd.DataFrame, left_on: str = 'name', right_on: str = 'name'):
    """
    Compare two dataframes for testing purposes
    
    Args:
        left: First dataframe
        right: Second dataframe
        left_on: Join column in left dataframe
        right_on: Join column in right dataframe
        
    Returns:
        Tuple of (merged dataframe, mismatches dataframe)
    """
    merged = left.merge(right, how='outer', left_on=left_on, right_on=right_on, indicator=True)

    print(f'Merge indicator column: {merged._merge.value_counts()}')

    columns_left = {c for c in left.columns if c != left_on}
    columns_right = {c for c in right.columns if c != right_on}

    print(f'Columns left only: {columns_left - columns_right}')
    print(f'Columns right only: {columns_right - columns_left}')

    mismatches = dict()

    for c in columns_left.intersection(columns_right):

        c_left = f'{c}_x'
        c_right = f'{c}_y'

        if merged[c_left].dtype != merged[c_right].dtype:

            print(f'Column {c} has dtype {merged[c_left].dtype} on left, dtype {merged[c_right].dtype} on right.')

        if is_numeric_dtype(merged[c_left]) and is_numeric_dtype(merged[c_right]):

            equalities = np.isclose(merged[c_left], merged[c_right], rtol=1e-05, equal_nan=True)

        else:

            equalities = (merged[c_left] == merged[c_right])

        # If a row is only in one dataframe, we don't count it as a mismatch - we're reported row discrepancy
        # already and we're now looking for inequality.
        equalities = equalities | (merged._merge != 'both')

        mismatches[c] = len(equalities) - equalities.sum()

    mismatches = pd.DataFrame.from_dict(data=mismatches, orient='index', columns=['mismatches'])

    return merged.sort_index(axis=1), mismatches


# Backwards compatibility: Keep old function names
get_spark_session = get_dask_client
