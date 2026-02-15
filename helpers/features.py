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
    
    # Combine
    df = dd.concat([df_out, df_in], axis=0)

    # 'caller_id' contains the subscriber in question for featurization purposes
    df = filter_by_phone_numbers_to_featurize(phone_numbers_to_featurize, df, 'caller_id')

    # Assign interactions to conversations if relevant
    df = tag_conversations(df)
    
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

    out = df.groupby(['caller_id', 'weekday', 'daytime'])['day'].nunique().reset_index(name='active_days')

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['active_days'],
                   indicator_name='active_days')

    return out


def number_of_contacts(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the number of distinct contacts per user, disaggregated by type and time of day, and transaction type
    """
    df = add_all_cat(df, cols='week_day')

    out = df.groupby(['caller_id', 'weekday', 'daytime', 'txn_type'])['recipient_id'].nunique().reset_index(name='number_of_contacts')

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'], values=['number_of_contacts'],
                   indicator_name='number_of_contacts')

    return out


def call_duration(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns summary stats of users' call durations, disaggregated by type and time of day
    """
    df = df[df['txn_type'] == 'call']
    df = add_all_cat(df, cols='week_day')

    stats_dict = summary_stats('duration')
    out = df.groupby(['caller_id', 'weekday', 'daytime', 'txn_type']).agg(stats_dict).reset_index()
    
    # Flatten column names
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
    out = df.groupby(['caller_id', 'weekday', 'txn_type'])['nocturnal'].mean().reset_index(name='percent_nocturnal')

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
    
    out = df_conv_start.groupby(['caller_id', 'weekday', 'daytime'])['initiated'].mean().reset_index(name='percent_initiated_conversations')
    
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
    out = df.groupby(['caller_id', 'weekday', 'daytime'])['initiated'].mean().reset_index(name='percent_initiated_interactions')

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['percent_initiated_interactions'],
                   indicator_name='percent_initiated_interactions')

    return out


def response_delay_text(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns summary stats of users' delays in responding to texts, disaggregated by type and time of day
    """
    df = df[df['txn_type'] == 'text']
    df = add_all_cat(df, cols='week_day')

    # Sort for lag operation
    df = df.sort_values(['caller_id', 'recipient_id', 'conversation', 'timestamp'])
    
    # Calculate previous direction and response delay
    df['prev_dir'] = df.groupby(['caller_id', 'recipient_id', 'conversation'])['direction'].shift(1)
    df['response_delay'] = np.where(
        (df['direction'] == 'out') & (df['prev_dir'] == 'in'),
        df['wait'],
        np.nan
    )
    
    stats_dict = summary_stats('response_delay')
    out = df.groupby(['caller_id', 'weekday', 'daytime']).agg(stats_dict).reset_index()
    out.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in out.columns.values]

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='response_delay_text')

    return out


def response_rate_text(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the percentage of texts to which the users responded, disaggregated by type and time of day
    """
    df = df[df['txn_type'] == 'text']
    df = add_all_cat(df, cols='week_day')

    # Check if user responded in each conversation
    df['dir_out'] = (df['direction'] == 'out').astype(int)
    df['responded'] = df.groupby(['caller_id', 'recipient_id', 'conversation'])['dir_out'].transform('max')
    
    # Filter to conversation starts incoming
    df_conv = df[
        (df['conversation'] == df['timestamp'].astype('int64') // 10**9) &
        (df['direction'] == 'in')
    ]
    
    out = df_conv.groupby(['caller_id', 'weekday', 'daytime'])['responded'].mean().reset_index(name='response_rate_text')

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
    counts = df.groupby(['caller_id', 'recipient_id', 'weekday', 'daytime', 'txn_type']).size().reset_index(name='n')
    
    # Calculate total per user group
    totals = counts.groupby(['caller_id', 'weekday', 'daytime', 'txn_type'])['n'].sum().reset_index(name='n_total')
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
    disaggregated by type and time of day, and transaction type
    """
    df = add_all_cat(df, cols='week_day')
    
    # Count by direction
    counts = df.groupby(['caller_id', 'recipient_id', 'direction', 'weekday', 'daytime', 'txn_type']).size().reset_index(name='n')
    
    # Pivot directions
    counts_pivot = counts.pivot_table(
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
    
    # Calculate summary stats
    stats_dict = summary_stats('n')
    out = counts_pivot.groupby(['caller_id', 'weekday', 'daytime', 'txn_type']).agg(stats_dict).reset_index()
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

    counts = df.groupby(['caller_id', 'recipient_id', 'weekday', 'daytime', 'txn_type']).size().reset_index(name='n')
    
    stats_dict = summary_stats('n')
    out = counts.groupby(['caller_id', 'weekday', 'daytime', 'txn_type']).agg(stats_dict).reset_index()
    out.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in out.columns.values]

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='interactions_per_contact')

    return out


def interevent_time(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns summary stats for the time between users' interactions, disaggregated by type and time of day, and
    transaction type
    """
    df = add_all_cat(df, cols='week_day')

    # Sort and calculate wait times
    df = df.sort_values(['caller_id', 'weekday', 'daytime', 'txn_type', 'timestamp'])
    df['ts'] = df['timestamp'].astype('int64') // 10**9
    df['prev_ts'] = df.groupby(['caller_id', 'weekday', 'daytime', 'txn_type'])['ts'].shift(1)
    df['wait'] = df['ts'] - df['prev_ts']
    
    stats_dict = summary_stats('wait')
    out = df.groupby(['caller_id', 'weekday', 'daytime', 'txn_type']).agg(stats_dict).reset_index()
    out.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in out.columns.values]

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
    counts = df.groupby(['caller_id', 'recipient_id', 'weekday', 'daytime', 'txn_type']).size().reset_index(name='n')
    
    # Compute to pandas for complex operations
    counts_pd = counts.compute()
    
    # Sort and calculate cumulative sums
    counts_pd = counts_pd.sort_values(['caller_id', 'weekday', 'daytime', 'txn_type', 'n'], ascending=[True, True, True, True, False])
    counts_pd['total'] = counts_pd.groupby(['caller_id', 'weekday', 'daytime', 'txn_type'])['n'].transform('sum')
    counts_pd['cumsum'] = counts_pd.groupby(['caller_id', 'weekday', 'daytime', 'txn_type'])['n'].cumsum()
    counts_pd['fraction'] = counts_pd['cumsum'] / counts_pd['total']
    counts_pd['row_number'] = counts_pd.groupby(['caller_id', 'weekday', 'daytime', 'txn_type']).cumcount() + 1
    
    # Find pareto threshold
    pareto_users = counts_pd[counts_pd['fraction'] >= percentage].groupby(['caller_id', 'weekday', 'daytime', 'txn_type'])['row_number'].min().reset_index(name='pareto_users')
    n_users = counts_pd.groupby(['caller_id', 'weekday', 'daytime', 'txn_type'])['recipient_id'].nunique().reset_index(name='n_users')
    
    out = pareto_users.merge(n_users, on=['caller_id', 'weekday', 'daytime', 'txn_type'])
    out['pareto'] = out['pareto_users'] / out['n_users']
    
    # Convert back to Dask
    out = dd.from_pandas(out, npartitions=4)

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
    pareto_users = durations_pd[durations_pd['fraction'] >= percentage].groupby(['caller_id', 'weekday', 'daytime'])['row_number'].min().reset_index(name='pareto_users')
    n_users = durations_pd.groupby(['caller_id', 'weekday', 'daytime'])['recipient_id'].nunique().reset_index(name='n_users')
    
    out = pareto_users.merge(n_users, on=['caller_id', 'weekday', 'daytime'])
    out['pareto'] = out['pareto_users'] / out['n_users']
    
    # Convert back to Dask
    out = dd.from_pandas(out, npartitions=4)

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['pareto'],
                   indicator_name='percent_pareto_durations')

    return out


def number_of_interactions(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the number of interactions per user, disaggregated by type and time of day, transaction type, and direction
    """
    df = add_all_cat(df, cols='week_day_dir')

    out = df.groupby(['caller_id', 'weekday', 'daytime', 'txn_type', 'direction']).size().reset_index(name='n')

    out = pivot_df(out, index=['caller_id'], columns=['direction', 'weekday', 'daytime', 'txn_type'], values=['n'],
                   indicator_name='number_of_interactions')

    return out


def number_of_antennas(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the number of antennas that handled users' interactions, disaggregated by type and time of day
    """
    df = add_all_cat(df, cols='week_day')

    out = df.groupby(['caller_id', 'weekday', 'daytime'])['caller_antenna'].nunique().reset_index(name='n_antennas')

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['n_antennas'],
                   indicator_name='number_of_antennas')

    return out


def entropy_of_antennas(df: DaskDataFrame) -> DaskDataFrame:
    """
    Returns the entropy of a user's antennas' shares of handled interactions, disaggregated by type and time of day
    """
    df = add_all_cat(df, cols='week_day')

    # Count interactions per antenna
    counts = df.groupby(['caller_id', 'caller_antenna', 'weekday', 'daytime']).size().reset_index(name='n')
    
    # Calculate totals
    totals = counts.groupby(['caller_id', 'weekday', 'daytime'])['n'].sum().reset_index(name='n_total')
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

    # Compute home antennas for all users (most frequent night antenna)
    night_counts = df[df['daytime'] == 'night'].groupby(['caller_id', 'caller_antenna']).size().reset_index(name='n')
    night_counts = night_counts.sort_values(['caller_id', 'n'], ascending=[True, False])
    home_antenna = night_counts.groupby('caller_id').first().reset_index()[['caller_id', 'caller_antenna']]
    home_antenna = home_antenna.rename(columns={'caller_antenna': 'home_antenna'})

    # Join with main dataframe
    df = df.merge(home_antenna, on='caller_id', how='inner')
    df['home_interaction'] = (df['caller_antenna'] == df['home_antenna']).astype(int)
    
    out = df.groupby(['caller_id', 'weekday', 'daytime'])['home_interaction'].mean().reset_index(name='mean')

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

    # Join with antennas
    df = df.merge(antennas, left_on='caller_antenna', right_on='antenna_id', how='inner')
    df = df.dropna(subset=['latitude', 'longitude'])

    # Calculate barycenter (weighted center)
    bar = df.groupby(['caller_id', 'weekday', 'daytime']).agg({
        'latitude': 'sum',
        'longitude': 'sum'
    }).reset_index()
    bar['n'] = df.groupby(['caller_id', 'weekday', 'daytime']).size().reset_index(name='n')['n']
    bar['bar_lat'] = bar['latitude'] / bar['n']
    bar['bar_lon'] = bar['longitude'] / bar['n']
    bar = bar.drop(columns=['latitude', 'longitude'])

    # Merge barycenter back
    df = df.merge(bar, on=['caller_id', 'weekday', 'daytime'])
    
    # Calculate great circle distance
    df = great_circle_distance(df)
    
    # Calculate radius of gyration
    out = df.groupby(['caller_id', 'weekday', 'daytime']).apply(
        lambda x: np.sqrt(np.sum(x['r']**2 / x['n'].iloc[0])),
        meta=('r', 'float64')
    ).reset_index()
    out.columns = ['caller_id', 'weekday', 'daytime', 'r']

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
    counts = df.groupby(['caller_id', 'caller_antenna', 'weekday', 'daytime']).size().reset_index(name='n')
    
    # Compute to pandas for complex operations
    counts_pd = counts.compute()
    
    # Sort and calculate cumulative sums
    counts_pd = counts_pd.sort_values(['caller_id', 'weekday', 'daytime', 'n'], ascending=[True, True, True, False])
    counts_pd['total'] = counts_pd.groupby(['caller_id', 'weekday', 'daytime'])['n'].transform('sum')
    counts_pd['cumsum'] = counts_pd.groupby(['caller_id', 'weekday', 'daytime'])['n'].cumsum()
    counts_pd['fraction'] = counts_pd['cumsum'] / counts_pd['total']
    counts_pd['row_number'] = counts_pd.groupby(['caller_id', 'weekday', 'daytime']).cumcount() + 1
    
    # Find pareto threshold
    out = counts_pd[counts_pd['fraction'] >= percentage].groupby(['caller_id', 'weekday', 'daytime'])['row_number'].min().reset_index(name='pareto_antennas')
    
    # Convert back to Dask
    out = dd.from_pandas(out, npartitions=4)

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['pareto_antennas'],
                   indicator_name='frequent_antennas')

    return out
