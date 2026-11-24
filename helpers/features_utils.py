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

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from itertools import product

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

    # Duplicate rows so pivots include every combination of "all" buckets:
    #   week_day: (allweek, allday) requires BOTH weekday=allweek AND daytime=allday
    #   on the same rows — not just separate copies that fix one column at a time.
    # Using itertools.product({False, True}^k) yields 2^k blocks: original, each
    # single-"all" slice, …, full (allweek, allday[, alldir]) joint category.
    keys = list(col_mapping.keys())
    vals = list(col_mapping.values())

    def _expand(partition, keys, vals):
        copies = []
        for flags in product([False, True], repeat=len(keys)):
            copy = partition.copy()
            for i, use_all in enumerate(flags):
                if use_all:
                    copy[keys[i]] = vals[i]
            copies.append(copy)
        return pd.concat(copies, ignore_index=True)

    return df.map_partitions(_expand, keys, vals)


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
    # Compute to pandas. Use synchronous scheduler to avoid repartitiontofewer tasks
    # in the distributed scheduler, which kill workers via OOM. With the temp CDR
    # base parquet cache (64 partitions), sync compute here is fast (seconds, not
    # hours as it was when reading the 1407 raw CDR files each time).
    # DO NOT call len(df) — in dask-expr that triggers taskshuffle with split_out=1
    # (nunique default) → all shuffled data goes to ONE worker → OOM.
    if isinstance(df, dd.DataFrame):
        with dask.config.set(scheduler='synchronous'):
            df_pandas = df.compute()
    else:
        df_pandas = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)

    if len(df_pandas) == 0:
        return dd.from_pandas(pd.DataFrame({'caller_id': pd.Series(dtype='str')}), npartitions=1)
    
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
    
    # Prefix indicator, but avoid active_days_active_days_… when the pivot value
    # column already starts with indicator_name (pandas includes the value level).
    rename_dict = {}
    for col in pivot_table.columns:
        if col == 'caller_id':
            continue
        c = str(col)
        if c.startswith(f'{indicator_name}_'):
            rename_dict[col] = c
        else:
            rename_dict[col] = f'{indicator_name}_{c}'
    pivot_table = pivot_table.rename(columns=rename_dict)
    
    # Convert back to Dask DataFrame
    result = dd.from_pandas(pivot_table)
    
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
    # Normalize key dtypes before any shuffle/sort to prevent mixed int/str partition failures.
    df = df.assign(
        caller_id=df['caller_id'].astype(str),
        recipient_id=df['recipient_id'].astype(str),
    )

    # Avoid df.sort_values() — Dask's distributed sort creates repartitiontofewer tasks
    # internally.  When a worker is paused at 80% memory those tasks deadlock
    # indefinitely (the repartitiontofewer task is assigned to the paused worker but
    # can never run because the worker is blocked waiting to spill).
    #
    # Fix: two-step approach.
    # 1. shuffle() by (caller_id, recipient_id) using the tasks method (hash-partition):
    #    all rows for the same caller/recipient pair end up in the same partition.
    #    Tasks-based shuffle does NOT create repartitiontofewer — it hash-routes rows
    #    without first determining global divisions.
    # 2. map_partitions() local sort by (caller_id, recipient_id, timestamp):
    #    each partition is already self-contained so the window logic below is correct.
    df = df.shuffle(['caller_id', 'recipient_id'])
    df = df.map_partitions(
        lambda p: p.sort_values(['caller_id', 'recipient_id', 'timestamp']).reset_index(drop=True)
    )

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
        
        # Clean up temporary columns (keep 'wait' - used by interevent_time and others)
        partition_df = partition_df.drop(columns=['ts', 'prev_txn', 'prev_ts', 'convo'], errors='ignore')
        
        return partition_df
    
    # Apply to each partition; meta must include all output columns in same order as partition returns (wait then conversation)
    meta = df._meta.copy()
    meta['wait'] = np.float64(0.0)
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
    Standard list of aggregation functions to be applied to column after group by (pandas NamedAgg format).
    """
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


def groupby_agg_summary_stats_dask(
    df: DaskDataFrame, group_cols: List[str], value_col: str
) -> DaskDataFrame:
    """
    Run groupby().agg(summary_stats(...)) in a Dask-compatible way.
    Dask's groupby.agg does not support the pandas NamedAgg (tuple) format, so we compute
    the needed columns to pandas, run the agg, then wrap back in Dask.
    Returns DataFrame with group_cols + mean, std, min, max, median, skewness, kurtosis.
    """
    with dask.config.set(scheduler='synchronous'):
        subset = df[group_cols + [value_col]].compute()
    # Use NamedAgg keyword form so output columns are mean, min, max, std, median, skewness, kurtosis
    out = subset.groupby(group_cols).agg(
        mean=(value_col, 'mean'),
        min=(value_col, 'min'),
        max=(value_col, 'max'),
        std=(value_col, 'std'),
        median=(value_col, lambda x: x.quantile(0.5)),
        skewness=(value_col, lambda x: x.skew()),
        kurtosis=(value_col, lambda x: x.kurtosis()),
    ).reset_index()
    return dd.from_pandas(out)
