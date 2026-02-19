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

from typing import Optional, List
import numpy as np
import pandas as pd
import dask.dataframe as dd

from box import Box
from helpers.features_utils import *
from helpers.utils import filter_by_phone_numbers_to_featurize

# Type alias
DaskDataFrame = dd.DataFrame


def all_dask(
    df: DaskDataFrame,
    antennas: DaskDataFrame,
    cfg: Box,
    phone_numbers_to_featurize: Optional[DaskDataFrame]
) -> List[DaskDataFrame]:
    """
    Compute cdr features starting from raw interaction data

    Args:
        df: dask dataframe with cdr interactions
        antennas: dask dataframe with antenna ids and coordinates
        cfg: config object
        phone_numbers_to_featurize: optional dataframe of phone numbers to featurize

    Returns:
        features: list of features as dask dataframes
    """
    features = []
    df_input = df.copy()
    # Partitioning column for Dask shuffle (merge and sort_values) is caller_id: set_partitions_pre
    # in shuffle.py uses it for divisions.searchsorted(); int/str mix across partitions raises TypeError.
    df = df.assign(
        caller_id=df['caller_id'].astype(str),
        recipient_id=df['recipient_id'].astype(str),
    )

    # Add weekday and daytime columns for subsequent groupby(s)
    df['weekday'] = df['day'].dt.dayofweek.apply(
        lambda x: 'weekend' if x in cfg.weekend else 'weekday',
        meta=('weekday', 'object')
    )
    df['daytime'] = df['timestamp'].dt.hour.apply(
        lambda x: 'night' if (x < cfg.start_of_day) or (x >= cfg.end_of_day) else 'day',
        meta=('daytime', 'object')
    )
    
    # Duplicate rows, switching caller and recipient columns (for both directions)
    # Create outgoing dataframe
    df_out = df.copy()
    df_out['direction'] = 'out'
    
    # Create incoming dataframe
    df_in = df.copy()
    df_in['direction'] = 'in'
    df_in['caller_id_temp'] = df_in['caller_id']
    df_in['caller_antenna_temp'] = df_in['caller_antenna']
    df_in['caller_id'] = df_in['recipient_id']
    df_in['recipient_id'] = df_in['caller_id_temp']
    df_in['caller_antenna'] = df_in['recipient_antenna']
    df_in['recipient_antenna'] = df_in['caller_antenna_temp']
    df_in = df_in.drop(columns=['caller_id_temp', 'caller_antenna_temp'])
    
    # Combine (ignore_index=True to avoid duplicate labels; downstream assigns/compute require unique index)
    df = dd.concat([df_out, df_in], axis=0, ignore_index=True)

    # 'caller_id' contains the subscriber in question for featurization purposes
    df = filter_by_phone_numbers_to_featurize(phone_numbers_to_featurize, df, 'caller_id')
    df = df.assign(
        caller_id=df['caller_id'].astype(str),
        recipient_id=df['recipient_id'].astype(str),
    )

    # Assign interactions to conversations if relevant
    df = tag_conversations(df)
    df = df.assign(
        caller_id=df['caller_id'].astype(str),
        recipient_id=df['recipient_id'].astype(str),
    )
    
    # Compute features and append them to list
    features.append(active_days(df))
    features.append(number_of_contacts(df))
    features.append(call_duration(df))
    features.append(percent_nocturnal(df))
    features.append(percent_initiated_conversations(df))
    features.append(percent_initiated_interactions(df))
    features.append(response_delay_text(df))
    features.append(response_rate_text(df))
    features.append(entropy_of_contacts(df))
    features.append(balance_of_contacts(df))
    features.append(interactions_per_contact(df))
    features.append(interevent_time(df))
    features.append(percent_pareto_interactions(df))
    features.append(percent_pareto_durations(df))
    features.append(number_of_interactions(df))
    features.append(number_of_antennas(df))
    features.append(entropy_of_antennas(df))
    features.append(radius_of_gyration(df, antennas))
    features.append(frequent_antennas(df))
    features.append(percent_at_home(df))

    return features


# Keep backward compatibility
all_spark = all_dask


def active_days(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the number of active days per user, disaggregated by type and time of day
    """
    df = add_all_cat(df, cols='week_day')

    out = df.groupby(['caller_id', 'weekday', 'daytime'])['day'].nunique().rename('active_days').reset_index()

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['active_days'],
                   indicator_name='active_days')

    return out


def number_of_contacts(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the number of distinct contacts per user, disaggregated by type and time of day, and transaction type
    """
    df = add_all_cat(df, cols='week_day')

    out = df.groupby(['caller_id', 'weekday', 'daytime', 'txn_type'])['recipient_id'].nunique().rename('number_of_contacts').reset_index()

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'], values=['number_of_contacts'],
                   indicator_name='number_of_contacts')

    return out


def call_duration(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns summary stats of users' call durations, disaggregated by type and time of day
    """
    df = df[df['txn_type'] == 'call']
    df = add_all_cat(df, cols='week_day')

    out = groupby_agg_summary_stats_dask(
        df, ['caller_id', 'weekday', 'daytime', 'txn_type'], 'duration'
    )
    if hasattr(out.columns, 'levels'):  # MultiIndex
        out.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in out.columns.values]

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='call_duration')

    return out


def percent_nocturnal(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the percentage of interactions done at night, per user, disaggregated by type of day and transaction type
    """
    df = add_all_cat(df, cols='week')

    df['nocturnal'] = (df['daytime'] == 'night').astype(int)
    out = df.groupby(['caller_id', 'weekday', 'txn_type'])['nocturnal'].mean().rename('percent_nocturnal').reset_index()

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'txn_type'], values=['percent_nocturnal'],
                   indicator_name='percent_nocturnal')

    return out


def percent_initiated_conversations(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the percentage of conversations initiated by the user, disaggregated by type and time of day
    """
    df = add_all_cat(df, cols='week_day')

    # Filter to conversation starts (where conversation timestamp equals message timestamp)
    df_conv_start = df[df['conversation'] == df['timestamp'].astype('int64') // 10**9]
    df_conv_start['initiated'] = (df_conv_start['direction'] == 'out').astype(int)
    
    out = df_conv_start.groupby(['caller_id', 'weekday', 'daytime'])['initiated'].mean().rename('percent_initiated_conversations').reset_index()
    
    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['percent_initiated_conversations'],
                   indicator_name='percent_initiated_conversations')

    return out


def percent_initiated_interactions(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the percentage of interactions initiated by the user, disaggregated by type and time of day
    """
    df = df[df['txn_type'] == 'call']
    df = add_all_cat(df, cols='week_day')

    df['initiated'] = (df['direction'] == 'out').astype(int)
    out = df.groupby(['caller_id', 'weekday', 'daytime'])['initiated'].mean().rename('percent_initiated_interactions').reset_index()

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['percent_initiated_interactions'],
                   indicator_name='percent_initiated_interactions')

    return out


def response_delay_text(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns summary stats of users' delays in responding to texts, disaggregated by type and time of day.
    Computes to pandas before adding response_delay to avoid Dask assign/reindex on duplicate labels.
    """
    df = df[df['txn_type'] == 'text']
    df = add_all_cat(df, cols='week_day')
    # Compute only needed columns (no response_delay/prev_dir) so no assign in graph → no duplicate-label reindex
    cols_to_load = ['caller_id', 'recipient_id', 'conversation', 'timestamp', 'direction', 'wait', 'weekday', 'daytime']
    subset = df[cols_to_load].compute()
    subset['caller_id'] = subset['caller_id'].astype(str)
    subset['recipient_id'] = subset['recipient_id'].astype(str)
    # Rest in pandas
    subset = subset.sort_values(['caller_id', 'recipient_id', 'conversation', 'timestamp'])
    subset['prev_dir'] = subset.groupby(['caller_id', 'recipient_id', 'conversation'])['direction'].shift(1)
    cond = (subset['direction'] == 'out') & (subset['prev_dir'] == 'in')
    subset['response_delay'] = subset['wait'].where(cond)
    out = subset.groupby(['caller_id', 'weekday', 'daytime']).agg(
        mean=('response_delay', 'mean'),
        min=('response_delay', 'min'),
        max=('response_delay', 'max'),
        std=('response_delay', 'std'),
        median=('response_delay', lambda x: x.quantile(0.5)),
        skewness=('response_delay', lambda x: x.skew()),
        kurtosis=('response_delay', lambda x: x.kurtosis()),
    ).reset_index()
    out = dd.from_pandas(out)
    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='response_delay_text')
    return out


def response_rate_text(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the percentage of texts to which the users responded, disaggregated by type and time of day.
    Computes to pandas before adding dir_out/responded to avoid Dask assign/reindex on duplicate labels.
    """
    df = df[df['txn_type'] == 'text']
    df = add_all_cat(df, cols='week_day')
    # Compute only needed columns (no dir_out/responded) so no assign in graph
    cols_to_load = ['caller_id', 'recipient_id', 'conversation', 'timestamp', 'direction', 'weekday', 'daytime']
    subset = df[cols_to_load].compute()
    # Rest in pandas
    subset['dir_out'] = (subset['direction'] == 'out').astype(int)
    subset['responded'] = subset.groupby(['caller_id', 'recipient_id', 'conversation'])['dir_out'].transform('max')
    conv_start = (subset['conversation'] == subset['timestamp'].astype('int64') // 10**9) & (subset['direction'] == 'in')
    df_conv = subset[conv_start]
    out = df_conv.groupby(['caller_id', 'weekday', 'daytime'])['responded'].mean().rename('response_rate_text').reset_index()
    out = dd.from_pandas(out)
    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['response_rate_text'],
                   indicator_name='response_rate_text')
    return out


def entropy_of_contacts(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the entropy of interactions the users had with their contacts, disaggregated by type and time of day, and
    transaction type
    """
    df = add_all_cat(df, cols='week_day')

    # Count interactions per contact
    counts = df.groupby(['caller_id', 'recipient_id', 'weekday', 'daytime', 'txn_type']).size().rename('n').reset_index()
    
    # Calculate total per user group
    totals = counts.groupby(['caller_id', 'weekday', 'daytime', 'txn_type'])['n'].sum().rename('n_total').reset_index()
    counts = counts.merge(totals, on=['caller_id', 'weekday', 'daytime', 'txn_type'])
    
    # Calculate proportions and entropy
    counts['n_prop'] = counts['n'] / counts['n_total']
    counts['entropy_term'] = counts['n_prop'] * np.log(counts['n_prop'])
    
    out = counts.groupby(['caller_id', 'weekday', 'daytime', 'txn_type'])['entropy_term'].sum().reset_index()
    out['entropy'] = -1 * out['entropy_term']
    out = out.drop(columns=['entropy_term'])

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'], values=['entropy'],
                   indicator_name='entropy_of_contacts')

    return out


def balance_of_contacts(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns summary stats for the balance of interactions (out/(in+out)) the users had with their contacts,
    disaggregated by type and time of day, and transaction type.
    Pivot is done in pandas because Dask pivot_table only accepts a single index column.
    """
    df = add_all_cat(df, cols='week_day')

    # Count by direction
    counts = df.groupby(['caller_id', 'recipient_id', 'direction', 'weekday', 'daytime', 'txn_type']).size().rename('n').reset_index()

    # Pivot in pandas (Dask pivot_table requires single column for index)
    counts_pd = counts.compute()
    counts_pivot = counts_pd.pivot_table(
        index=['caller_id', 'recipient_id', 'weekday', 'daytime', 'txn_type'],
        columns='direction',
        values='n',
        fill_value=0
    ).reset_index()

    # Ensure both in and out columns exist
    for direction in ['in', 'out']:
        if direction not in counts_pivot.columns:
            counts_pivot[direction] = 0

    # Calculate balance
    counts_pivot['n_total'] = counts_pivot['in'] + counts_pivot['out']
    counts_pivot['n'] = counts_pivot['out'] / counts_pivot['n_total']

    # Summary stats (counts_pivot is pandas; use pandas agg then wrap)
    out = groupby_agg_summary_stats_dask(
        dd.from_pandas(counts_pivot), ['caller_id', 'weekday', 'daytime', 'txn_type'], 'n'
    )
    if hasattr(out.columns, 'levels'):  # MultiIndex
        out.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in out.columns.values]

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='balance_of_contacts')

    return out


def interactions_per_contact(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns summary stats for the number of interactions the users had with their contacts, disaggregated by type and
    time of day, and transaction type
    """
    df = add_all_cat(df, cols='week_day')

    counts = df.groupby(['caller_id', 'recipient_id', 'weekday', 'daytime', 'txn_type']).size().rename('n').reset_index()

    out = groupby_agg_summary_stats_dask(
        counts, ['caller_id', 'weekday', 'daytime', 'txn_type'], 'n'
    )
    if hasattr(out.columns, 'levels'):  # MultiIndex
        out.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in out.columns.values]

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='interactions_per_contact')

    return out


def interevent_time(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns summary stats for the time between users' interactions, disaggregated by type and time of day, and
    transaction type. Computes to pandas before adding ts/prev_ts/wait to avoid Dask assign/reindex on duplicate labels.
    """
    df = add_all_cat(df, cols='week_day')
    # Compute only needed columns (no ts, prev_ts, wait) so no assign in graph
    cols_to_load = ['caller_id', 'weekday', 'daytime', 'txn_type', 'timestamp']
    subset = df[cols_to_load].compute()
    subset['caller_id'] = subset['caller_id'].astype(str)
    # Rest in pandas
    subset = subset.sort_values(['caller_id', 'weekday', 'daytime', 'txn_type', 'timestamp'])
    subset['ts'] = subset['timestamp'].astype('int64') // 10**9
    subset['prev_ts'] = subset.groupby(['caller_id', 'weekday', 'daytime', 'txn_type'])['ts'].shift(1)
    subset['wait'] = subset['ts'] - subset['prev_ts']
    out = subset.groupby(['caller_id', 'weekday', 'daytime', 'txn_type']).agg(
        mean=('wait', 'mean'),
        min=('wait', 'min'),
        max=('wait', 'max'),
        std=('wait', 'std'),
        median=('wait', lambda x: x.quantile(0.5)),
        skewness=('wait', lambda x: x.skew()),
        kurtosis=('wait', lambda x: x.kurtosis()),
    ).reset_index()
    out = dd.from_pandas(out)
    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='interevent_time')
    return out


def percent_pareto_interactions(df: DaskDataFrame, percentage: float = 0.8) -> DaskDataFrame:
    """
    Returns the percentage of a user's contacts that account for 80% of their interactions, disaggregated by type and
    time of day, and transaction type
    
    Note: This requires compute() for complex window operations
    """
    df = add_all_cat(df, cols='week_day')

    # Count interactions per contact
    counts = df.groupby(['caller_id', 'recipient_id', 'weekday', 'daytime', 'txn_type']).size().rename('n').reset_index()
    
    # Compute to pandas for complex operations
    counts_pd = counts.compute()
    
    # Sort and calculate cumulative sums
    counts_pd = counts_pd.sort_values(['caller_id', 'weekday', 'daytime', 'txn_type', 'n'], ascending=[True, True, True, True, False])
    counts_pd['total'] = counts_pd.groupby(['caller_id', 'weekday', 'daytime', 'txn_type'])['n'].transform('sum')
    counts_pd['cumsum'] = counts_pd.groupby(['caller_id', 'weekday', 'daytime', 'txn_type'])['n'].cumsum()
    counts_pd['fraction'] = counts_pd['cumsum'] / counts_pd['total']
    counts_pd['row_number'] = counts_pd.groupby(['caller_id', 'weekday', 'daytime', 'txn_type']).cumcount() + 1
    
    # Find pareto threshold
    pareto_users = counts_pd[counts_pd['fraction'] >= percentage].groupby(['caller_id', 'weekday', 'daytime', 'txn_type'])['row_number'].min().rename('pareto_users').reset_index()
    n_users = counts_pd.groupby(['caller_id', 'weekday', 'daytime', 'txn_type'])['recipient_id'].nunique().rename('n_users').reset_index()
    
    out = pareto_users.merge(n_users, on=['caller_id', 'weekday', 'daytime', 'txn_type'])
    out['pareto'] = out['pareto_users'] / out['n_users']
    
    # Convert back to Dask
    out = dd.from_pandas(out)

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'], values=['pareto'],
                   indicator_name='percent_pareto_interactions')

    return out


def percent_pareto_durations(df: DaskDataFrame, percentage: float = 0.8) -> DaskDataFrame:
    """
    Returns the percentage of a user's contacts that account for 80% of their call durations, disaggregated by type and
    time of day, and transaction type
    
    Note: This requires compute() for complex window operations
    """
    df = df[df['txn_type'] == 'call']
    df = add_all_cat(df, cols='week_day')

    # Sum durations per contact
    durations = df.groupby(['caller_id', 'recipient_id', 'weekday', 'daytime'])['duration'].sum().reset_index()
    
    # Compute to pandas for complex operations
    durations_pd = durations.compute()
    
    # Sort and calculate cumulative sums
    durations_pd = durations_pd.sort_values(['caller_id', 'weekday', 'daytime', 'duration'], ascending=[True, True, True, False])
    durations_pd['total'] = durations_pd.groupby(['caller_id', 'weekday', 'daytime'])['duration'].transform('sum')
    durations_pd['cumsum'] = durations_pd.groupby(['caller_id', 'weekday', 'daytime'])['duration'].cumsum()
    durations_pd['fraction'] = durations_pd['cumsum'] / durations_pd['total']
    durations_pd['row_number'] = durations_pd.groupby(['caller_id', 'weekday', 'daytime']).cumcount() + 1
    
    # Find pareto threshold
    pareto_users = durations_pd[durations_pd['fraction'] >= percentage].groupby(['caller_id', 'weekday', 'daytime'])['row_number'].min().rename('pareto_users').reset_index()
    n_users = durations_pd.groupby(['caller_id', 'weekday', 'daytime'])['recipient_id'].nunique().rename('n_users').reset_index()
    
    out = pareto_users.merge(n_users, on=['caller_id', 'weekday', 'daytime'])
    out['pareto'] = out['pareto_users'] / out['n_users']
    
    # Convert back to Dask
    out = dd.from_pandas(out)

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['pareto'],
                   indicator_name='percent_pareto_durations')

    return out


def number_of_interactions(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the number of interactions per user, disaggregated by type and time of day, transaction type, and direction
    """
    df = add_all_cat(df, cols='week_day_dir')

    out = df.groupby(['caller_id', 'weekday', 'daytime', 'txn_type', 'direction']).size().rename('n').reset_index()

    out = pivot_df(out, index=['caller_id'], columns=['direction', 'weekday', 'daytime', 'txn_type'], values=['n'],
                   indicator_name='number_of_interactions')

    return out


def number_of_antennas(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the number of antennas that handled users' interactions, disaggregated by type and time of day
    """
    df = add_all_cat(df, cols='week_day')

    out = df.groupby(['caller_id', 'weekday', 'daytime'])['caller_antenna'].nunique().rename('n_antennas').reset_index()

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['n_antennas'],
                   indicator_name='number_of_antennas')

    return out


def entropy_of_antennas(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the entropy of a user's antennas' shares of handled interactions, disaggregated by type and time of day
    """
    df = add_all_cat(df, cols='week_day')

    # Count interactions per antenna
    counts = df.groupby(['caller_id', 'caller_antenna', 'weekday', 'daytime']).size().rename('n').reset_index()
    
    # Calculate totals
    totals = counts.groupby(['caller_id', 'weekday', 'daytime'])['n'].sum().rename('n_total').reset_index()
    counts = counts.merge(totals, on=['caller_id', 'weekday', 'daytime'])
    
    # Calculate entropy
    counts['n_prop'] = counts['n'] / counts['n_total']
    counts['entropy_term'] = counts['n_prop'] * np.log(counts['n_prop'])
    
    out = counts.groupby(['caller_id', 'weekday', 'daytime'])['entropy_term'].sum().reset_index()
    out['entropy'] = -1 * out['entropy_term']
    out = out.drop(columns=['entropy_term'])

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['entropy'],
                   indicator_name='entropy_of_antennas')

    return out


def percent_at_home(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the percentage of interactions handled by a user's home antenna, disaggregated by type and time of day
    """
    df = add_all_cat(df, cols='week_day')

    df = df.dropna(subset=['caller_antenna'])

    # Compute home antennas for all users (most frequent night antenna).
    # Do ranking in pandas after explicit key normalization to avoid mixed-type
    # set_partitions_pre shuffle failures in Dask sort_values.
    night_counts = (
        df[df['daytime'] == 'night']
        .groupby(['caller_id', 'caller_antenna'])
        .size()
        .rename('n')
        .reset_index()
        .assign(
            caller_id=lambda x: x['caller_id'].astype(str),
            caller_antenna=lambda x: x['caller_antenna'].astype(str),
        )
    )
    night_counts_pd = night_counts.compute()
    if night_counts_pd.empty:
        return dd.from_pandas(pd.DataFrame({'caller_id': pd.Series(dtype='str')}), npartitions=1)

    night_counts_pd = night_counts_pd.sort_values(['caller_id', 'n'], ascending=[True, False])
    home_antenna_pd = (
        night_counts_pd
        .drop_duplicates(subset=['caller_id'], keep='first')[['caller_id', 'caller_antenna']]
        .rename(columns={'caller_antenna': 'home_antenna'})
    )
    home_antenna = dd.from_pandas(home_antenna_pd)

    # Join with main dataframe
    df = df.assign(
        caller_id=df['caller_id'].astype(str),
        caller_antenna=df['caller_antenna'].astype(str),
    )
    df = df.merge(home_antenna, on='caller_id', how='inner')
    df['home_interaction'] = (df['caller_antenna'] == df['home_antenna']).astype(int)
    
    out = df.groupby(['caller_id', 'weekday', 'daytime'])['home_interaction'].mean().rename('mean').reset_index()

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['mean'],
                   indicator_name='percent_at_home')

    return out


def radius_of_gyration(df: DaskDataFrame, antennas: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the radius of gyration of users, disaggregated by type and time of day

    References
    ----------
    .. [GON2008] Gonzalez, M. C., Hidalgo, C. A., & Barabasi, A. L. (2008).
        Understanding individual human mobility patterns. Nature, 453(7196),
        779-782.
    """
    df = add_all_cat(df, cols='week_day')

    # Cast merge keys to same type (CDR may have float, antennas may have string)
    df = df.assign(caller_antenna=df['caller_antenna'].astype(str))
    antennas_str = antennas.assign(antenna_id=antennas['antenna_id'].astype(str))
    df = df.merge(antennas_str, left_on='caller_antenna', right_on='antenna_id', how='inner')
    df = df.dropna(subset=['latitude', 'longitude'])

    # Calculate barycenter (weighted center)
    bar = df.groupby(['caller_id', 'weekday', 'daytime']).agg({
        'latitude': 'sum',
        'longitude': 'sum'
    }).reset_index()
    bar['n'] = df.groupby(['caller_id', 'weekday', 'daytime']).size().rename('n').reset_index()['n']
    bar['bar_lat'] = bar['latitude'] / bar['n']
    bar['bar_lon'] = bar['longitude'] / bar['n']
    bar = bar.drop(columns=['latitude', 'longitude'])

    # Merge barycenter back
    df = df.merge(bar, on=['caller_id', 'weekday', 'daytime'])
    
    # Calculate great circle distance
    df = great_circle_distance(df)

    # Radius of gyration: do groupby/apply in pandas so result has [caller_id, weekday, daytime, r] for pivot_df
    cols_for_rog = ['caller_id', 'weekday', 'daytime', 'r', 'n']
    subset = df[cols_for_rog].compute()
    out = subset.groupby(['caller_id', 'weekday', 'daytime']).apply(
        lambda x: np.sqrt(np.sum(x['r']**2 / x['n'].iloc[0]))
    ).reset_index()
    # pandas may produce 4 or 5 columns from apply+reset_index; keep first 3 as keys and last as 'r'
    out = out.iloc[:, [0, 1, 2, -1]].copy()
    out.columns = ['caller_id', 'weekday', 'daytime', 'r']
    out = dd.from_pandas(out)
    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['r'],
                   indicator_name='radius_of_gyration')
    return out


def frequent_antennas(df: DaskDataFrame, percentage: float = 0.8) -> DaskDataFrame:
    """
    Returns the percentage of antennas accounting for 80% of users' interactions, disaggregated by type and time of day
    
    Note: This requires compute() for complex window operations
    """
    df = add_all_cat(df, cols='week_day')

    # Count interactions per antenna
    counts = df.groupby(['caller_id', 'caller_antenna', 'weekday', 'daytime']).size().rename('n').reset_index()
    
    # Compute to pandas for complex operations
    counts_pd = counts.compute()
    
    # Sort and calculate cumulative sums
    counts_pd = counts_pd.sort_values(['caller_id', 'weekday', 'daytime', 'n'], ascending=[True, True, True, False])
    counts_pd['total'] = counts_pd.groupby(['caller_id', 'weekday', 'daytime'])['n'].transform('sum')
    counts_pd['cumsum'] = counts_pd.groupby(['caller_id', 'weekday', 'daytime'])['n'].cumsum()
    counts_pd['fraction'] = counts_pd['cumsum'] / counts_pd['total']
    counts_pd['row_number'] = counts_pd.groupby(['caller_id', 'weekday', 'daytime']).cumcount() + 1
    
    # Find pareto threshold
    out = counts_pd[counts_pd['fraction'] >= percentage].groupby(['caller_id', 'weekday', 'daytime'])['row_number'].min().rename('pareto_antennas').reset_index()
    
    # Convert back to Dask
    out = dd.from_pandas(out)

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['pareto_antennas'],
                   indicator_name='frequent_antennas')

    return out
