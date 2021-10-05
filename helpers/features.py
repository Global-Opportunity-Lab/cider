from box import Box
from helpers.features_utils import *


def all_spark(df: SparkDataFrame, antennas: SparkDataFrame, cfg: Box) -> List[SparkDataFrame]:
    """
    Compute cdr features starting from raw interaction data

    Args:
        df: spark dataframe with cdr interactions
        antennas: spark dataframe with antenna ids and coordinates
        cfg: config object

    Returns:
        features: list of features as spark dataframes
    """
    features = []

    df = (df
          # Add weekday and daytime columns for subsequent groupby(s)
          .withColumn('weekday', F.when(F.dayofweek('day').isin(cfg.weekend), 'weekend').otherwise('weekday'))
          .withColumn('daytime', F.when((F.hour('timestamp') < cfg.start_of_day) |
                                        (F.hour('timestamp') >= cfg.end_of_day), 'night').otherwise('day'))
          # Duplicate rows, switching caller and recipient columns
          .withColumn('direction', lit('out'))
          .withColumn('directions', F.array(lit('in'), col('direction')))
          .withColumn('direction', F.explode('directions'))
          .withColumn('caller_id_copy', col('caller_id'))
          .withColumn('caller_antenna_copy', col('caller_antenna'))
          .withColumn('caller_id', F.when(col('direction') == 'in', col('recipient_id')).otherwise(col('caller_id')))
          .withColumn('recipient_id',
                      F.when(col('direction') == 'in', col('caller_id_copy')).otherwise(col('recipient_id')))
          .withColumn('caller_antenna',
                      F.when(col('direction') == 'in', col('recipient_antenna')).otherwise(col('caller_antenna')))
          .withColumn('recipient_antenna',
                      F.when(col('direction') == 'in', col('caller_antenna_copy')).otherwise(col('recipient_antenna')))
          .drop('directions', 'caller_id_copy', 'recipient_antenna_copy'))

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
    features.append((balance_of_contacts(df)))
    features.append(interactions_per_contact(df))
    features.append(interevent_time(df))
    features.append(percent_pareto_interactions(df))
    features.append((percent_pareto_durations(df)))
    features.append(number_of_interactions(df))
    features.append(number_of_antennas(df))
    features.append(entropy_of_antennas(df))
    features.append(radius_of_gyration(df, antennas))
    features.append(frequent_antennas(df))
    features.append(percent_at_home(df))

    return features


def active_days(df: SparkDataFrame) -> SparkDataFrame:
    """
    Returns the number of active days per user, disaggregated by type and time of day
    """
    df = add_all_cat(df, cols='week_day')

    out = (df
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.countDistinct('day').alias('active_days')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['active_days'],
                   indicator_name='active_days')

    return out


def number_of_contacts(df: SparkDataFrame) -> SparkDataFrame:
    """
    Returns the number of distinct contacts per user, disaggregated by type and time of day, and transaction type
    """
    df = add_all_cat(df, cols='week_day')

    out = (df
           .groupby('caller_id', 'weekday', 'daytime', 'txn_type')
           .agg(F.countDistinct('recipient_id').alias('number_of_contacts')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'], values=['number_of_contacts'],
                   indicator_name='number_of_contacts')

    return out


def call_duration(df: SparkDataFrame) -> SparkDataFrame:
    """
    Returns summary stats of users' call durations, disaggregated by type and time of day
    """
    df = df.where(col('txn_type') == 'call')
    df = add_all_cat(df, cols='week_day')

    out = (df
           .groupby('caller_id', 'weekday', 'daytime', 'txn_type')
           .agg(*summary_stats('duration')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='call_duration')

    return out


def percent_nocturnal(df: SparkDataFrame) -> SparkDataFrame:
    """
    Returns the percentage of interactions done at night, per user, disaggregated by type of day and transaction type
    """
    df = add_all_cat(df, cols='week')

    out = (df
           .withColumn('nocturnal', F.when(col('daytime') == 'night', 1).otherwise(0))
           .groupby('caller_id', 'weekday', 'txn_type')
           .agg(F.mean('nocturnal').alias('percent_nocturnal')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'txn_type'], values=['percent_nocturnal'],
                   indicator_name='percent_nocturnal')

    return out


def percent_initiated_conversations(df: SparkDataFrame) -> SparkDataFrame:
    """
    Returns the percentage of conversations initiated by the user, disaggregated by type and time of day
    """
    df = add_all_cat(df, cols='week_day')

    out = (df
           .where(col('conversation') == col('timestamp').cast('long'))
           .withColumn('initiated', F.when(col('direction') == 'out', 1).otherwise(0))
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.mean('initiated').alias('percent_initiated_conversations')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['percent_initiated_conversations'],
                   indicator_name='percent_initiated_conversations')

    return out


def percent_initiated_interactions(df: SparkDataFrame) -> SparkDataFrame:
    """
    Returns the percentage of interactions initiated by the user, disaggregated by type and time of day
    """
    df = df.where(col('txn_type') == 'call')
    df = add_all_cat(df, cols='week_day')

    out = (df
           .withColumn('initiated', F.when(col('direction') == 'out', 1).otherwise(0))
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.mean('initiated').alias('percent_initiated_interactions')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['percent_initiated_interactions'],
                   indicator_name='percent_initiated_interactions')

    return out


def response_delay_text(df: SparkDataFrame) -> SparkDataFrame:
    """
    Returns summary stats of users' delays in responding to texts, disaggregated by type and time of day
    """
    df = df.where(col('txn_type') == 'text')
    df = add_all_cat(df, cols='week_day')

    w = Window.partitionBy('caller_id', 'recipient_id', 'conversation').orderBy('timestamp')
    out = (df
           .withColumn('prev_dir', F.lag(col('direction')).over(w))
           .withColumn('response_delay', F.when((col('direction') == 'out') & (col('prev_dir') == 'in'), col('wait')))
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(*summary_stats('response_delay')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='response_delay_text')

    return out


def response_rate_text(df: SparkDataFrame) -> SparkDataFrame:
    """
    Returns the percentage of texts to which the users responded, disaggregated by type and time of day
    """
    df = df.where(col('txn_type') == 'text')
    df = add_all_cat(df, cols='week_day')

    w = Window.partitionBy('caller_id', 'recipient_id', 'conversation')
    out = (df
           .withColumn('dir', F.when(col('direction') == 'out', 1).otherwise(0))
           .withColumn('responded', F.max(col('dir')).over(w))
           .where((col('conversation') == col('timestamp').cast('long')) & (col('direction') == 'in'))
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.mean('responded').alias('response_rate_text')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['response_rate_text'],
                   indicator_name='response_rate_text')

    return out


def entropy_of_contacts(df: SparkDataFrame) -> SparkDataFrame:
    """
    Returns the entropy of interactions the users had with their contacts, disaggregated by type and time of day, and
    transaction type
    """
    df = add_all_cat(df, cols='week_day')

    w = Window.partitionBy('caller_id', 'weekday', 'daytime', 'txn_type')
    out = (df
           .groupby('caller_id', 'recipient_id', 'weekday', 'daytime', 'txn_type')
           .agg(F.count(lit(0)).alias('n'))
           .withColumn('n_total', F.sum('n').over(w))
           .withColumn('n', (col('n')/col('n_total').cast('float')))
           .groupby('caller_id', 'weekday', 'daytime', 'txn_type')
           .agg((-1*F.sum(col('n')*F.log(col('n')))).alias('entropy')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'], values=['entropy'],
                   indicator_name='entropy_of_contacts')

    return out


def balance_of_contacts(df: SparkDataFrame) -> SparkDataFrame:
    """
    Returns summary stats for the balance of interactions (out/(in+out)) the users had with their contacts,
    disaggregated by type and time of day, and transaction type
    """
    df = add_all_cat(df, cols='week_day')

    out = (df
           .groupby('caller_id', 'recipient_id', 'direction', 'weekday', 'daytime', 'txn_type')
           .agg(F.count(lit(0)).alias('n'))
           .groupby('caller_id', 'recipient_id', 'weekday', 'daytime', 'txn_type')
           .pivot('direction')
           .agg(F.first('n').alias('n'))
           .fillna(0)
           .withColumn('n_total', col('in')+col('out'))
           .withColumn('n', (col('out')/col('n_total')))
           .groupby('caller_id', 'weekday', 'daytime', 'txn_type')
           .agg(*summary_stats('n')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='balance_of_contacts')

    return out


def interactions_per_contact(df: SparkDataFrame) -> SparkDataFrame:
    """
    Returns summary stats for the number of interactions the users had with their contacts, disaggregated by type and
    time of day, and transaction type
    """
    df = add_all_cat(df, cols='week_day')

    out = (df
           .groupby('caller_id', 'recipient_id', 'weekday', 'daytime', 'txn_type')
           .agg(F.count(lit(0)).alias('n'))
           .groupby('caller_id', 'weekday', 'daytime', 'txn_type')
           .agg(*summary_stats('n')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='interactions_per_contact')

    return out


def interevent_time(df: SparkDataFrame) -> SparkDataFrame:
    """
    Returns summary stats for the time between users' interactions, disaggregated by type and time of day, and
    transaction type
    """
    df = add_all_cat(df, cols='week_day')

    w = Window.partitionBy('caller_id', 'weekday', 'daytime', 'txn_type').orderBy('timestamp')
    out = (df
           .withColumn('ts', col('timestamp').cast('long'))
           .withColumn('prev_ts', F.lag(col('ts')).over(w))
           .withColumn('wait', col('ts') - col('prev_ts'))
           .groupby('caller_id', 'weekday', 'daytime', 'txn_type')
           .agg(*summary_stats('wait')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='interevent_time')

    return out


def percent_pareto_interactions(df: SparkDataFrame, percentage: float = 0.8) -> SparkDataFrame:
    """
    Returns the percentage of a user's contacts that account for 80% of their interactions, disaggregated by type and
    time of day, and transaction type
    """
    df = add_all_cat(df, cols='week_day')

    w = Window.partitionBy('caller_id', 'weekday', 'daytime', 'txn_type')
    w1 = Window.partitionBy('caller_id', 'weekday', 'daytime', 'txn_type').orderBy(col('n').desc())
    w2 = Window.partitionBy('caller_id', 'weekday', 'daytime', 'txn_type').orderBy('row_number')
    out = (df
           .groupby('caller_id', 'recipient_id', 'weekday', 'daytime', 'txn_type')
           .agg(F.count(lit(0)).alias('n'))
           .withColumn('row_number', F.row_number().over(w1))
           .withColumn('total', F.sum('n').over(w))
           .withColumn('cumsum', F.sum('n').over(w2))
           .withColumn('fraction', col('cumsum')/col('total'))
           .withColumn('row_number', F.when(col('fraction') >= percentage, col('row_number')))
           .groupby('caller_id', 'weekday', 'daytime', 'txn_type')
           .agg(F.min('row_number').alias('pareto_users'),
                F.countDistinct('recipient_id').alias('n_users'))
           .withColumn('pareto', col('pareto_users')/col('n_users')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'], values=['pareto'],
                   indicator_name='percent_pareto_interactions')

    return out


def percent_pareto_durations(df: SparkDataFrame, percentage: float = 0.8) -> SparkDataFrame:
    """
    Returns the percentage of a user's contacts that account for 80% of their call durations, disaggregated by type and
    time of day, and transaction type
    """
    df = df.where(col('txn_type') == 'call')
    df = add_all_cat(df, cols='week_day')

    w = Window.partitionBy('caller_id', 'weekday', 'daytime')
    w1 = Window.partitionBy('caller_id', 'weekday', 'daytime').orderBy(col('duration').desc())
    w2 = Window.partitionBy('caller_id', 'weekday', 'daytime').orderBy('row_number')
    out = (df
           .groupby('caller_id', 'recipient_id', 'weekday', 'daytime')
           .agg(F.sum('duration').alias('duration'))
           .withColumn('row_number', F.row_number().over(w1))
           .withColumn('total', F.sum('duration').over(w))
           .withColumn('cumsum', F.sum('duration').over(w2))
           .withColumn('fraction', col('cumsum')/col('total'))
           .withColumn('row_number', F.when(col('fraction') >= percentage, col('row_number')))
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.min('row_number').alias('pareto_users'),
                F.countDistinct('recipient_id').alias('n_users'))
           .withColumn('pareto', col('pareto_users')/col('n_users')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['pareto'],
                   indicator_name='percent_pareto_durations')

    return out


def number_of_interactions(df: SparkDataFrame) -> SparkDataFrame:
    """
    Returns the number of interactions per user, disaggregated by type and time of day, transaction type, and direction
    """
    df = add_all_cat(df, cols='week_day_dir')

    out = (df
           .groupby('caller_id', 'weekday', 'daytime', 'txn_type', 'direction')
           .agg(F.count(lit(0)).alias('n')))

    out = pivot_df(out, index=['caller_id'], columns=['direction', 'weekday', 'daytime', 'txn_type'], values=['n'],
                   indicator_name='number_of_interactions')

    return out


def number_of_antennas(df: SparkDataFrame) -> SparkDataFrame:
    """
    Returns the number of antennas the handled users' interactions, disaggregated by type and time of day
    """
    df = add_all_cat(df, cols='week_day')

    out = (df
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.countDistinct('caller_antenna').alias('n_antennas')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['n_antennas'],
                   indicator_name='number_of_antennas')

    return out


def entropy_of_antennas(df: SparkDataFrame) -> SparkDataFrame:
    """
    Returns the entropy of a user's antennas' shares of handled interactions, disaggregated by type and time of day
    """
    df = add_all_cat(df, cols='week_day')

    w = Window.partitionBy('caller_id', 'weekday', 'daytime')
    out = (df
           .groupby('caller_id', 'caller_antenna', 'weekday', 'daytime')
           .agg(F.count(lit(0)).alias('n'))
           .withColumn('n_total', F.sum('n').over(w))
           .withColumn('n', (col('n')/col('n_total').cast('float')))
           .groupby('caller_id', 'weekday', 'daytime')
           .agg((-1*F.sum(col('n')*F.log(col('n')))).alias('entropy')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['entropy'],
                   indicator_name='entropy_of_antennas')

    return out


def percent_at_home(df: SparkDataFrame) -> SparkDataFrame:
    """
    Returns the percentage of interactions handled by a user's home antenna, disaggregated by type and time of day
    """
    df = add_all_cat(df, cols='week_day')

    df = df.dropna(subset=['caller_antenna'])

    # Compute home antennas for all users, if possible
    w = Window.partitionBy('caller_id').orderBy(col('n').desc())
    home_antenna = (df
                    .where(col('daytime') == 'night')
                    .groupby('caller_id', 'caller_antenna')
                    .agg(F.count(lit(0)).alias('n'))
                    .withColumn('row_number', F.row_number().over(w))
                    .where(col('row_number') == 1)
                    .withColumnRenamed('caller_antenna', 'home_antenna')
                    .drop('n'))

    out = (df
           .join(home_antenna, on='caller_id', how='inner')
           .withColumn('home_interaction', F.when(col('caller_antenna') == col('home_antenna'), 1).otherwise(0))
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.mean('home_interaction').alias('mean')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['mean'],
                   indicator_name='percent_at_home')

    return out


def radius_of_gyration(df: SparkDataFrame, antennas: SparkDataFrame) -> SparkDataFrame:
    """
    Returns the radius of gyration of users, disaggregated by type and time of day

    References
    ----------
    .. [GON2008] Gonzalez, M. C., Hidalgo, C. A., & Barabasi, A. L. (2008).
        Understanding individual human mobility patterns. Nature, 453(7196),
        779-782.
    """
    df = add_all_cat(df, cols='week_day')

    df = (df
          .join(antennas, on=df.caller_antenna == antennas.antenna_id, how='inner')
          .dropna(subset=['latitude', 'longitude']))

    bar = (df
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.sum('latitude').alias('latitude'),
                F.sum('longitude').alias('longitude'),
                F.count(lit(0)).alias('n'))
           .withColumn('bar_lat', col('latitude')/col('n'))
           .withColumn('bar_lon', col('longitude') / col('n'))
           .drop('latitude', 'longitude'))

    df = df.join(bar, on=['caller_id', 'weekday', 'daytime'])
    df = great_circle_distance(df)
    out = (df
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.sqrt(F.sum(col('r')**2/col('n'))).alias('r')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['r'],
                   indicator_name='radius_of_gyration')

    return out


def frequent_antennas(df: SparkDataFrame, percentage: float = 0.8) -> SparkDataFrame:
    """
    Returns the percentage of antennas accounting for 80% of users' interactions, disaggregated by type and time of day
    """
    df = add_all_cat(df, cols='week_day')

    w = Window.partitionBy('caller_id', 'weekday', 'daytime')
    w1 = Window.partitionBy('caller_id', 'weekday', 'daytime').orderBy(col('n').desc())
    w2 = Window.partitionBy('caller_id', 'weekday', 'daytime').orderBy('row_number')
    out = (df
           .groupby('caller_id', 'caller_antenna', 'weekday', 'daytime')
           .agg(F.count(lit(0)).alias('n'))
           .withColumn('row_number', F.row_number().over(w1))
           .withColumn('total', F.sum('n').over(w))
           .withColumn('cumsum', F.sum('n').over(w2))
           .withColumn('fraction', col('cumsum')/col('total'))
           .withColumn('row_number', F.when(col('fraction') >= percentage, col('row_number')))
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.min('row_number').alias('pareto_antennas')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['pareto_antennas'],
                   indicator_name='frequent_antennas')

    return out
