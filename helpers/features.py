import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit
from pyspark.sql.window import Window


def all_spark(df):
    features = []

    df = (df
          .withColumn('weekday', F.when(F.dayofweek('day').isin([1, 7]), 'weekend').otherwise('weekday'))
          .withColumn('daytime', F.when((F.hour('timestamp') < 7)|(F.hour('timestamp') >= 19), 'night').otherwise('day'))
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

    df = tag_conversations(df)

    #features.append(active_days(df))
    #features.append(number_of_contacts(df))
    #features.append(call_duration(df))
    #features.append(percent_nocturnal(df))
    #features.append(percent_initiated_conversations(df))
    #features.append(percent_initiated_interactions(df))
    #features.append(response_delay_text(df))
    #features.append(response_rate_text(df))
    #features.append(entropy_of_contacts(df))
    #features.append((balance_of_contacts(df)))
    #features.append(interactions_per_contact(df))
    #features.append(interevent_time(df))
    #features.append(percent_pareto_interactions(df))
    #features.append((percent_pareto_durations(df)))
    #features.append(number_of_interactions(df))
    #features.append(number_of_antennas(df))
    features.append(entropy_of_antennas(df))

    return features


def active_days(df):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

    out = (df
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.countDistinct('day').alias('active_days')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['active_days'],
                   indicator_name='active_days')

    return out


def number_of_contacts(df):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

    out = (df
           .groupby('caller_id', 'weekday', 'daytime', 'txn_type')
           .agg(F.countDistinct('recipient_id').alias('number_of_contacts')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'], values=['number_of_contacts'],
                   indicator_name='number_of_contacts')

    return out


def call_duration(df):
    df = df.where(col('txn_type') == 'call')
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

    out = (df
           .groupby('caller_id', 'weekday', 'daytime', 'txn_type')
           .agg(F.mean('duration').alias('mean'),
                F.min('duration').alias('min'),
                F.max('duration').alias('max'),
                F.stddev_pop('duration').alias('std'),
                F.expr('percentile_approx(duration, 0.5)').alias('median'),
                F.skewness('duration').alias('skewness'),
                F.kurtosis('duration').alias('kurtosis')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='call_duration')

    return out


def percent_nocturnal(df):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek'})

    out = (df
           .withColumn('nocturnal', F.when(col('daytime') == 'night', 1).otherwise(0))
           .groupby('caller_id', 'weekday', 'txn_type')
           .agg(F.mean('nocturnal').alias('percent_nocturnal')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'txn_type'], values=['percent_nocturnal'],
                   indicator_name='percent_nocturnal')

    return out


def percent_initiated_conversations(df):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

    out = (df
           .where(col('conversation') == col('timestamp').cast('long'))
           .withColumn('initiated', F.when(col('direction') == 'out', 1).otherwise(0))
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.mean('initiated').alias('percent_initiated_conversations')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['percent_initiated_conversations'],
                   indicator_name='percent_initiated_conversations')

    return out


def percent_initiated_interactions(df):
    df = df.where(col('txn_type') == 'call')
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

    out = (df
           .withColumn('initiated', F.when(col('direction') == 'out', 1).otherwise(0))
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.mean('initiated').alias('percent_initiated_interactions')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['percent_initiated_interactions'],
                   indicator_name='percent_initiated_interactions')

    return out


def response_delay_text(df):
    df = df.where(col('txn_type') == 'text')
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

    w = Window.partitionBy('caller_id', 'recipient_id', 'conversation').orderBy('timestamp')
    out = (df
           .withColumn('prev_dir', F.lag(col('direction')).over(w))
           .withColumn('response_delay', F.when((col('direction') == 'out')&(col('prev_dir') == 'in'), col('wait')))
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.mean('response_delay').alias('mean'),
                F.min('response_delay').alias('min'),
                F.max('response_delay').alias('max'),
                F.stddev_pop('response_delay').alias('std'),
                F.expr('percentile_approx(response_delay, 0.5)').alias('median'),
                F.skewness('response_delay').alias('skewness'),
                F.kurtosis('response_delay').alias('kurtosis')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='response_delay_text')

    return out


def response_rate_text(df):
    df = df.where(col('txn_type') == 'text')
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

    w = Window.partitionBy('caller_id', 'recipient_id', 'conversation').orderBy('timestamp')
    out = (df
           .withColumn('dir', F.when(col('direction') == 'out', 1).otherwise(0))
           .withColumn('responded', F.max(col('dir')).over(w))
           .where((col('conversation') == col('timestamp').cast('long')) & (col('direction') == 'in'))
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.mean('responded').alias('response_rate_text')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['response_rate_text'],
                   indicator_name='response_rate_text')

    return out


def entropy_of_contacts(df):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

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


def balance_of_contacts(df):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

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
           .agg(F.mean('n').alias('mean'),
                F.min('n').alias('min'),
                F.max('n').alias('max'),
                F.stddev_pop('n').alias('std'),
                F.expr('percentile_approx(n, 0.5)').alias('median'),
                F.skewness('n').alias('skewness'),
                F.kurtosis('n').alias('kurtosis')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='balance_of_contacts')

    return out


def interactions_per_contact(df):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

    out = (df
           .groupby('caller_id', 'recipient_id', 'weekday', 'daytime', 'txn_type')
           .agg(F.count(lit(0)).alias('n'))
           .groupby('caller_id', 'weekday', 'daytime', 'txn_type')
           .agg(F.mean('n').alias('mean'),
                F.min('n').alias('min'),
                F.max('n').alias('max'),
                F.stddev_pop('n').alias('std'),
                F.expr('percentile_approx(n, 0.5)').alias('median'),
                F.skewness('n').alias('skewness'),
                F.kurtosis('n').alias('kurtosis')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='interactions_per_contact')

    return out


def interevent_time(df):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

    w = Window.partitionBy('caller_id', 'weekday', 'daytime', 'txn_type').orderBy('timestamp')
    out = (df
           .withColumn('ts', col('timestamp').cast('long'))
           .withColumn('prev_ts', F.lag(col('ts')).over(w))
           .withColumn('wait', col('ts') - col('prev_ts'))
           .groupby('caller_id', 'weekday', 'daytime', 'txn_type')
           .agg(F.mean('wait').alias('mean'),
                F.min('wait').alias('min'),
                F.max('wait').alias('max'),
                F.stddev_pop('wait').alias('std'),
                F.expr('percentile_approx(wait, 0.5)').alias('median'),
                F.skewness('wait').alias('skewness'),
                F.kurtosis('wait').alias('kurtosis')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'],
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'],
                   indicator_name='interevent_time')

    return out


def percent_pareto_interactions(df, percentage=0.8):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

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


def percent_pareto_durations(df, percentage=0.8):
    df = df.where(col('txn_type') == 'call')
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

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


def number_of_interactions(df):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday',
                                      'direction': 'alldir'})

    out = (df
           .groupby('caller_id', 'weekday', 'daytime', 'txn_type', 'direction')
           .agg(F.count(lit(0)).alias('n')))

    out = pivot_df(out, index=['caller_id'], columns=['direction', 'weekday', 'daytime', 'txn_type'], values=['n'],
                   indicator_name='number_of_interactions')

    return out


def number_of_antennas(df):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday',})

    out = (df
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.countDistinct('caller_antenna').alias('n_antennas')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['n_antennas'],
                   indicator_name='number_of_antennas')

    return out


def entropy_of_antennas(df):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

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


def add_all_cat(df, col_mapping):
    for column, value in col_mapping.items():
        df = (df
              .withColumn(column, F.array(lit(value), col(column)))
              .withColumn(column, F.explode(column)))

    return df


def pivot_df(df, index, columns, values, indicator_name):
    while columns:
        column = columns.pop()
        df = (df
              .groupby(index + columns)
              .pivot(column)
              .agg(*[F.first(val).alias(val) for val in values]))
        values = [val for val in df.columns if val not in index and val not in columns]

    # Rename columns by prefixing indicator name
    col_selection = [col(col_name).alias(indicator_name + '_' + col_name) for col_name in df.columns if
                     col_name != 'caller_id']
    df = df.select('caller_id', *col_selection)

    return df


def tag_conversations(df):
    w = Window.partitionBy('caller_id', 'recipient_id').orderBy('timestamp')

    df = (df
          .withColumn('ts', col('timestamp').cast('long'))
          .withColumn('prev_txn', F.lag(col('txn_type')).over(w))
          .withColumn('prev_ts', F.lag(col('ts')).over(w))
          .withColumn('wait', col('ts') - col('prev_ts'))
          .withColumn('conversation', F.when((col('txn_type') == 'text')&
                                             ((col('prev_txn') == 'call')|
                                              (col('prev_txn').isNull())|
                                              (col('wait') >= 3600)), col('ts')))
          .withColumn('convo', F.last('conversation', ignorenulls=True).over(w))
          .withColumn('conversation', F.when(col('conversation').isNotNull(), col('conversation'))
                                       .otherwise(F.when(col('txn_type') == 'text', col('convo'))))
          .drop('ts', 'prev_txn', 'prev_ts', 'convo'))

    return df
