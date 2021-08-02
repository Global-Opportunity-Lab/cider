import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit
from pyspark.sql.window import Window


def add_all_cat(df, cols):
    if cols == 'week':
        col_mapping = {'weekday': 'allweek'}
    elif cols == 'week_day':
        col_mapping = {'weekday': 'allweek', 'daytime': 'allday'}
    elif cols == 'week_day_dir':
        col_mapping = {'weekday': 'allweek', 'daytime': 'allday', 'direction': 'alldir'}

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
    col_selection = [col(col_name).alias(indicator_name + '_' + col_name) for col_name in df.columns
                     if col_name != 'caller_id']
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


def summary_stats(col_name):
    functions = [
        F.mean(col_name).alias('mean'),
        F.min(col_name).alias('min'),
        F.max(col_name).alias('max'),
        F.stddev_pop(col_name).alias('std'),
        F.expr(f'percentile_approx({col_name}, 0.5)').alias('median'),
        F.skewness(col_name).alias('skewness'),
        F.kurtosis(col_name).alias('kurtosis')
    ]

    return functions


def great_circle_distance(df):
    r = 6371.

    df = (df
          .withColumn('delta_latitude', F.radians(col('latitude') - col('bar_lat')))
          .withColumn('delta_longitude', F.radians(col('longitude') - col('bar_lon')))
          .withColumn('latitude1', F.radians(col('latitude')))
          .withColumn('latitude2', F.radians(col('bar_lat')))
          .withColumn('a', F.sin(col('delta_latitude')/2)**2 +
                           F.cos('latitude1')*F.cos('latitude2')*(F.sin(col('delta_longitude')/2)**2))
          .withColumn('r', 2*lit(r)*F.asin(F.sqrt('a'))))

    return df
