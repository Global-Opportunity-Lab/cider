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

import dask.dataframe as dd
import pandas as pd
import numpy as np
from typing import List

# Type alias for Dask DataFrame
DaskDataFrame = dd.DataFrame


def add_all_cat(df: DaskDataFrame, cols: str) -> DaskDataFrame:
    """
    Duplicate dataframe rows so that groupby result includes an "all interactions" category for specified column(s)

    Args:
        df: dask dataframe
        cols: string defining columns to duplicate

    Returns:
        df: dask dataframe
    """
    # Define mapping from column name to column value
    # e.g. the column daytime will also have a value called "allday" to denote both day and night
    if cols == 'week':
        col_mapping = {'weekday': 'allweek'}
    elif cols == 'week_day':
        col_mapping = {'weekday': 'allweek', 'daytime': 'allday'}
    elif cols == 'week_day_dir':
        col_mapping = {'weekday': 'allweek', 'daytime': 'allday', 'direction': 'alldir'}
    else:
        raise ValueError("'cols' argument should be one of {week, week_day, week_day_dir}")

    # For each of the columns defined in the mapping, duplicate entries
    # In Dask, we create separate dataframes and concatenate them
    dfs_to_concat = [df]
    
    for column, value in col_mapping.items():
        # Create a copy with the "all" value for this column
        df_all = df.copy()
        df_all[column] = value
        dfs_to_concat.append(df_all)
    
    # Concatenate all dataframes
    result = dd.concat(dfs_to_concat, axis=0, ignore_index=True)
    
    return result


def pivot_df(df: DaskDataFrame,
             index: List[str], columns: List[str], values: List[str], indicator_name: str) -> DaskDataFrame:
    """
    Recreate pandas pivot method for dataframes
    
    Note: Complex pivots in Dask may require computing to pandas first for efficiency

    Args:
        df: dask dataframe
        index: columns to use to make new frame's index
        columns: columns to use to make new frame's columns
        values: column(s) to use for populating new frame's values
        indicator_name: name of indicator to prefix to new columns

    Returns:
        df: pivoted dask dataframe
    """
    # Check if dataframe is empty
    if len(df) == 0:
        return df[['caller_id']]

    # For complex multi-level pivots, compute to pandas for efficiency
    # Dask doesn't have native multi-level pivot support
    df_pandas = df.compute()
    
    # Build pivot table
    pivot_table = df_pandas.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc='first'
    )
    
    # Flatten column names
    if isinstance(pivot_table.columns, pd.MultiIndex):
        pivot_table.columns = ['_'.join(map(str, col)).strip() for col in pivot_table.columns.values]
    
    # Reset index to make index columns regular columns
    pivot_table = pivot_table.reset_index()
    
    # Rename columns by prefixing indicator name (except caller_id)
    rename_dict = {col: f'{indicator_name}_{col}' for col in pivot_table.columns if col != 'caller_id'}
    pivot_table = pivot_table.rename(columns=rename_dict)
    
    # Convert back to Dask DataFrame
    result = dd.from_pandas(pivot_table, npartitions=4)
    
    return result


def tag_conversations(df: DaskDataFrame, max_wait: int = 3600) -> DaskDataFrame:
    """
    From bandicoot's documentation: "We define conversations as a series of text messages between the user and one
    contact. A conversation starts with either of the parties sending a text to the other. A conversation will stop if
    no text was exchanged by the parties for an hour or if one of the parties call the other. The next conversation will
    start as soon as a new text is send by either of the parties."
    This functions tags interactions with the conversation id they are part of: the id is the start unix time of the
    conversation.

    Args:
        df: dask dataframe
        max_wait: time (in seconds) after which a conversation ends if no texts or calls have been exchanged

    Returns:
        df: tagged dask dataframe
    """
    # For window functions with complex logic, we need to work with partitions
    # Sort by the partition key columns and timestamp
    df = df.sort_values(['caller_id', 'recipient_id', 'timestamp'])
    
    def tag_conversations_partition(partition_df):
        """Apply conversation tagging to a partition"""
        if len(partition_df) == 0:
            return partition_df
        
        # Convert timestamp to unix timestamp
        partition_df['ts'] = partition_df['timestamp'].astype('int64') // 10**9
        
        # Calculate previous transaction type and timestamp
        partition_df['prev_txn'] = partition_df.groupby(['caller_id', 'recipient_id'])['txn_type'].shift(1)
        partition_df['prev_ts'] = partition_df.groupby(['caller_id', 'recipient_id'])['ts'].shift(1)
        
        # Calculate wait time
        partition_df['wait'] = partition_df['ts'] - partition_df['prev_ts']
        
        # Identify conversation starts
        partition_df['conversation'] = np.where(
            (partition_df['txn_type'] == 'text') &
            ((partition_df['prev_txn'] == 'call') |
             (partition_df['prev_txn'].isna()) |
             (partition_df['wait'] >= max_wait)),
            partition_df['ts'],
            np.nan
        )
        
        # Forward fill conversation IDs within each caller-recipient pair
        partition_df['convo'] = partition_df.groupby(['caller_id', 'recipient_id'])['conversation'].ffill()
        
        # Set conversation to convo for text messages without their own conversation start
        partition_df['conversation'] = np.where(
            partition_df['conversation'].notna(),
            partition_df['conversation'],
            np.where(partition_df['txn_type'] == 'text', partition_df['convo'], np.nan)
        )
        
        # Clean up temporary columns
        partition_df = partition_df.drop(columns=['ts', 'prev_txn', 'prev_ts', 'convo'], errors='ignore')
        
        return partition_df
    
    # Apply to each partition
    meta = df._meta.copy()
    meta['conversation'] = 0.0
    
    result = df.map_partitions(tag_conversations_partition, meta=meta)
    
    return result


def great_circle_distance(df: DaskDataFrame) -> DaskDataFrame:
    """
    Return the great-circle distance in kilometers between two points, in this case always the antenna handling an
    interaction and the barycenter of all the user's interactions.
    Used to compute the radius of gyration.
    """
    r = 6371.  # Earth's radius

    # Calculate differences in radians
    df['delta_latitude'] = np.radians(df['latitude'] - df['bar_lat'])
    df['delta_longitude'] = np.radians(df['longitude'] - df['bar_lon'])
    df['latitude1'] = np.radians(df['latitude'])
    df['latitude2'] = np.radians(df['bar_lat'])
    
    # Haversine formula
    df['a'] = (np.sin(df['delta_latitude']/2)**2 + 
               np.cos(df['latitude1']) * np.cos(df['latitude2']) * 
               (np.sin(df['delta_longitude']/2)**2))
    
    df['r'] = 2 * r * np.arcsin(np.sqrt(df['a']))
    
    return df


def summary_stats(col_name: str) -> dict:
    """
    Standard list of aggregation functions to be applied to column after group by
    
    Args:
        col_name: Name of the column to compute statistics for
        
    Returns:
        Dictionary mapping statistic names to aggregation functions
    """
    # In Dask, we return a dictionary for use with .agg()
    stats_dict = {
        f'{col_name}_mean': (col_name, 'mean'),
        f'{col_name}_min': (col_name, 'min'),
        f'{col_name}_max': (col_name, 'max'),
        f'{col_name}_std': (col_name, 'std'),
        f'{col_name}_median': (col_name, lambda x: x.quantile(0.5)),
        f'{col_name}_skewness': (col_name, lambda x: x.skew()),
        f'{col_name}_kurtosis': (col_name, lambda x: x.kurtosis())
    }
    
    return stats_dict
