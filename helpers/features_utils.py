from pyspark.sql import DataFrame as SparkDataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit
from pyspark.sql.window import Window
from typing import List


def add_all_cat(df: SparkDataFrame, cols: str) -> SparkDataFrame:
    """
    Duplicate dataframe rows so that groupby result includes an "all interactions" category for specified column(s)

    Args:
        df: spark dataframe
        cols: string defining columns to duplicate

    Returns:
        df: spark dataframe
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
    for column, value in col_mapping.items():
        df = (df
              .withColumn(column, F.array(lit(value), col(column)))
              .withColumn(column, F.explode(column)))

    return df


def pivot_df(df: SparkDataFrame,
             index: List[str], columns: List[str], values: List[str], indicator_name: str) -> SparkDataFrame:
    """
    Recreate pandas pivot method for dataframes

    Args:
        df: spark dataframe
        index: columns to use to make new frame’s index
        columns: columns to use to make new frame’s columns
        values: column(s) to use for populating new frame’s values
        indicator_name: name of indicator to prefix to new columns

    Returns:
        df: pivoted spark dataframe
    """
    # Iterate through columns
    while columns:
        column = columns.pop()
        # Pivot values based on current column selection
        df = (df
              .groupby(index + columns)
              .pivot(column)
              .agg(*[F.first(val).alias(val) for val in values]))
        #  Update values for next pivot
        values = [val for val in df.columns if val not in index and val not in columns]

    # Rename columns by prefixing indicator name
    col_selection = [col(col_name).alias(indicator_name + '_' + col_name) for col_name in df.columns
                     if col_name != 'caller_id']
    df = df.select('caller_id', *col_selection)

    return df


def tag_conversations(df: SparkDataFrame, max_wait: int = 3600) -> SparkDataFrame:
    """
    From bandicoot's documentation: "We define conversations as a series of text messages between the user and one
    contact. A conversation starts with either of the parties sending a text to the other. A conversation will stop if
    no text was exchanged by the parties for an hour or if one of the parties call the other. The next conversation will
    start as soon as a new text is send by either of the parties."
    This functions tags interactions with the conversation id they are part of: the id is the start unix time of the
    conversation.

    Args:
        df: spark dataframe
        max_wait: time (in seconds) after which a conversation ends if no texts or calls have been exchanged

    Returns:
        df: tagged spark dataframe
    """
    w = Window.partitionBy('caller_id', 'recipient_id').orderBy('timestamp')

    df = (df
          .withColumn('ts', col('timestamp').cast('long'))
          .withColumn('prev_txn', F.lag(col('txn_type')).over(w))
          .withColumn('prev_ts', F.lag(col('ts')).over(w))
          .withColumn('wait', col('ts') - col('prev_ts'))
          .withColumn('conversation', F.when((col('txn_type') == 'text') &
                                             ((col('prev_txn') == 'call') |
                                              (col('prev_txn').isNull()) |
                                              (col('wait') >= max_wait)), col('ts')))
          .withColumn('convo', F.last('conversation', ignorenulls=True).over(w))
          .withColumn('conversation', F.when(col('conversation').isNotNull(), col('conversation'))
                                       .otherwise(F.when(col('txn_type') == 'text', col('convo'))))
          .drop('ts', 'prev_txn', 'prev_ts', 'convo'))

    return df


def great_circle_distance(df: SparkDataFrame) -> SparkDataFrame:
    """
    Return the great-circle distance in kilometers between two points, in this case always the antenna handling an
    interaction and the barycenter of all the user's interactions.
    Used to compute the radius of gyration.
    """
    r = 6371.  # Earth's radius

    df = (df
          .withColumn('delta_latitude', F.radians(col('latitude') - col('bar_lat')))
          .withColumn('delta_longitude', F.radians(col('longitude') - col('bar_lon')))
          .withColumn('latitude1', F.radians(col('latitude')))
          .withColumn('latitude2', F.radians(col('bar_lat')))
          .withColumn('a', F.sin(col('delta_latitude')/2)**2 +
                      F.cos('latitude1')*F.cos('latitude2')*(F.sin(col('delta_longitude')/2)**2))
          .withColumn('r', 2*lit(r)*F.asin(F.sqrt('a'))))

    return df


def summary_stats(col_name: str) -> list:
    # Standard list of functions to be applied to column after group by
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
