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
          .withColumn('caller_id', F.when(col('direction') == 'in', col('recipient_id')).otherwise(col('caller_id')))
          .withColumn('recipient_id', F.when(col('direction') == 'in', col('caller_id_copy')).otherwise(col('recipient_id')))
          .drop('directions', 'caller_id_copy'))

    df = tag_conversations(df)

    #features.append(active_days(df))
    #features.append(number_of_contacts(df))
    #features.append(call_duration(df))
    #features.append(percent_nocturnal(df))
    #features.append(percent_initiated_conversations(df))
    #features.append(percent_initiated_interactions(df))
    #features.append(response_delay_text(df))
    features.append(response_rate_text(df))

    return features


def active_days(df):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

    out = (df
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.countDistinct('day').alias('active_days')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['active_days'])

    col_selection = [col(col_name).alias('active_days_' + col_name) for col_name in out.columns if col_name != 'caller_id']
    out = out.select('caller_id', *col_selection)

    return out


def number_of_contacts(df):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

    out = (df
           .groupby('caller_id', 'weekday', 'daytime', 'txn_type')
           .agg(F.countDistinct('recipient_id').alias('number_of_contacts')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime', 'txn_type'], values=['number_of_contacts'])

    col_selection = [col(col_name).alias('number_of_contacts_' + col_name) for col_name in out.columns if col_name != 'caller_id']
    out = out.select('caller_id', *col_selection)

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
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'])

    col_selection = [col(col_name).alias('call_duration_' + col_name) for col_name in out.columns if col_name != 'caller_id']
    out = out.select('caller_id', *col_selection)

    return out


def percent_nocturnal(df):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek'})

    out = (df
           .withColumn('nocturnal', F.when(col('daytime') == 'night', 1).otherwise(0))
           .groupby('caller_id', 'weekday', 'txn_type')
           .agg(F.mean('nocturnal').alias('percent_nocturnal')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'txn_type'], values=['percent_nocturnal'])

    col_selection = [col(col_name).alias('percent_nocturnal_' + col_name) for col_name in out.columns if col_name != 'caller_id']
    out = out.select('caller_id', *col_selection)

    return out


def percent_initiated_conversations(df):
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

    out = (df
           .where(col('conversation') == col('timestamp').cast('long'))
           .withColumn('initiated', F.when(col('direction') == 'out', 1).otherwise(0))
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.mean('initiated').alias('percent_initiated_conversations')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['percent_initiated_conversations'])

    col_selection = [col(col_name).alias('percent_initiated_conversations_' + col_name) for col_name in out.columns if col_name != 'caller_id']
    out = out.select('caller_id', *col_selection)

    return out


def percent_initiated_interactions(df):
    df = df.where(col('txn_type') == 'call')
    df = add_all_cat(df, col_mapping={'weekday': 'allweek',
                                      'daytime': 'allday'})

    out = (df
           .withColumn('initiated', F.when(col('direction') == 'out', 1).otherwise(0))
           .groupby('caller_id', 'weekday', 'daytime')
           .agg(F.mean('initiated').alias('percent_initiated_interactions')))

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['percent_initiated_interactions'])

    col_selection = [col(col_name).alias('percent_initiated_interactions_' + col_name) for col_name in out.columns if col_name != 'caller_id']
    out = out.select('caller_id', *col_selection)

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
                   values=['mean', 'std', 'median', 'skewness', 'kurtosis', 'min', 'max'])

    col_selection = [col(col_name).alias('response_delay_text_' + col_name) for col_name in out.columns if
                     col_name != 'caller_id']
    out = out.select('caller_id', *col_selection)

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

    out = pivot_df(out, index=['caller_id'], columns=['weekday', 'daytime'], values=['response_rate_text'])

    col_selection = [col(col_name).alias('response_rate_text_' + col_name) for col_name in out.columns if
                     col_name != 'caller_id']
    out = out.select('caller_id', *col_selection)

    return out


def add_all_cat(df, col_mapping):
    for column, value in col_mapping.items():
        df = (df
              .withColumn(column, F.array(lit(value), col(column)))
              .withColumn(column, F.explode(column)))

    return df


def pivot_df(df, index, columns, values):
    while columns:
        column = columns.pop()
        df = (df
              .groupby(index + columns)
              .pivot(column)
              .agg(*[F.first(val).alias(val) for val in values]))
        values = [val for val in df.columns if val not in index and val not in columns]

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
