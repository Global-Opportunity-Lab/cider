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

import pandas as pd
from datetime import datetime, timedelta
from typing import Any
from pyspark.sql import DataFrame as SparkDataFrame

from .schemas import (
    DataDiagnosticStatistics,
    AllowedPivotColumnsEnum,
    DirectionOfTransactionEnum,
    StatsComputationMethodEnum,
    MobileMoneyDataWithDay,
)
import pyspark.sql.functions as F
from pyspark.sql.functions import (
    col,
    when,
    lag,
    sum as pys_sum,
    mean as pys_mean,
    min as pys_min,
    max as pys_max,
    stddev_pop,
    expr,
    skewness,
    kurtosis,
    radians,
    sin as pys_sin,
    cos as pys_cos,
    lit,
    asin,
    sqrt,
    hour,
    dayofweek,
    last,
)
from pyspark.sql.window import Window
from cider.schemas import (
    CallDataRecordTransactionType,
    CallDataRecordData,
)
from cider.utils import validate_dataframe
import numpy as np


def _get_summary_stats_cols(
    col_name: str,
    summary_stats: list[StatsComputationMethodEnum] = [
        e for e in StatsComputationMethodEnum
    ],
) -> list:
    """
    Get summary statistics columns for a given column name.

    Args:
        col_name: name of the column for which to get summary statistics
        summary_stats: list of summary statistics to compute. Defaults to all statistics.

    Returns:
        list: list of summary statistics columns
    """
    summary_stats_mapping = {
        StatsComputationMethodEnum.MEAN: pys_mean,
        StatsComputationMethodEnum.MIN: pys_min,
        StatsComputationMethodEnum.MAX: pys_max,
        StatsComputationMethodEnum.STD: stddev_pop,
        StatsComputationMethodEnum.MEDIAN: expr(f"percentile_approx({col_name}, 0.5)"),
        StatsComputationMethodEnum.SKEWNESS: skewness,
        StatsComputationMethodEnum.KURTOSIS: kurtosis,
    }

    agg_stats = []

    for stat in summary_stats:
        agg_func = summary_stats_mapping.get(stat)
        if agg_func is None:
            continue
        if stat != StatsComputationMethodEnum.MEDIAN:
            agg_stats.append(agg_func(col_name).alias(f"{stat.value}_{col_name}"))
        else:
            agg_stats.append(agg_func.alias(f"{stat.value}_{col_name}"))

    return agg_stats


def _get_agg_columns_by_cdr_time_and_transaction_type(
    col_name: str,
    cols_to_use_for_pivot: list[AllowedPivotColumnsEnum],
    agg_func: F = pys_sum,
) -> list:
    """
    Get aggregation columns for pivoting CDR features based on transaction type, is_weekend and is_daytime.

    Args:
        col_name: name of the column to aggregate
        cols_to_use_for_pivot: list of columns to use for pivoting (e.g., ["is_weekend", "is_daytime", "transaction_type"])
        agg_func: aggregation function to use (default: sum)
    """

    # Pivot the dataframe to have separate columns for each combination
    is_weekday_values = [0, 1]
    is_daytime_values = [0, 1]
    transaction_types = [e.value for e in CallDataRecordTransactionType]

    meshgrid_values: list[Any] = []
    for pivot_col in cols_to_use_for_pivot:
        if pivot_col == AllowedPivotColumnsEnum.IS_WEEKEND:
            meshgrid_values.append(is_weekday_values)
        if pivot_col == AllowedPivotColumnsEnum.IS_DAYTIME:
            meshgrid_values.append(is_daytime_values)
        if pivot_col == AllowedPivotColumnsEnum.TRANSACTION_TYPE:
            meshgrid_values.append(transaction_types)

    meshgrid = (
        np.meshgrid(*meshgrid_values)
        if len(meshgrid_values) > 1
        else np.array([meshgrid_values])
    )
    meshgrid = np.array([m.flatten() for m in meshgrid]).T.squeeze()
    if meshgrid.ndim == 1:
        meshgrid = meshgrid.reshape(-1, 1)

    aggs = []
    for vals in meshgrid:
        agg_name = ""
        condition = True
        for i, pivot_col in enumerate(cols_to_use_for_pivot):
            if pivot_col == AllowedPivotColumnsEnum.IS_WEEKEND:
                is_weekend_val = vals[i]
                weekend_col = "weekend" if int(is_weekend_val) == 1 else "weekday"
                agg_name += f"{weekend_col}_"
                condition = condition & (col("is_weekend") == int(is_weekend_val))
            if pivot_col == AllowedPivotColumnsEnum.IS_DAYTIME:
                is_daytime_val = vals[i]
                daytime_col = "daytime" if int(is_daytime_val) == 1 else "nighttime"
                agg_name += f"{daytime_col}_"
                condition = condition & (col("is_daytime") == int(is_daytime_val))
            if pivot_col == AllowedPivotColumnsEnum.TRANSACTION_TYPE:
                transaction_type_val = vals[i]
                agg_name += f"{transaction_type_val}_"
                condition = condition & (
                    col("transaction_type") == transaction_type_val
                )
        agg_name += f"{col_name}"

        aggs.append(
            agg_func(when(condition, col(col_name)).otherwise(0.0)).alias(agg_name)
        )
    return aggs


def _great_circle_distance(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Return the great-circle distance in kilometers between two points, in this case always the antenna handling an
    interaction and the barycenter of all the user's interactions.
    Used to compute the radius of gyration.

    Args:
        spark_df: input spark dataframe with columns sum_latitude, sum_longitude,
                    center_of_mass_latitude, center_of_mass_longitude

    Returns:
        spark_df: spark dataframe with additional column 'radius' representing the great-circle distance
    """
    r = 6371.0  # Earth's radius

    spark_df = (
        spark_df.withColumn(
            "delta_latitude",
            radians(col("sum_latitude") - col("center_of_mass_latitude")),
        )
        .withColumn(
            "delta_longitude",
            radians(col("sum_longitude") - col("center_of_mass_longitude")),
        )
        .withColumn("latitude1", radians(col("sum_latitude")))
        .withColumn("latitude2", radians(col("center_of_mass_latitude")))
        .withColumn(
            "azimuth",
            pys_sin(col("delta_latitude") / 2) ** 2
            + pys_cos("latitude1")
            * pys_cos("latitude2")
            * (pys_sin(col("delta_longitude") / 2) ** 2),
        )
        .withColumn("radius", 2 * lit(r) * asin(sqrt("azimuth")))
    )

    return spark_df


def filter_to_datetime(
    df: pd.DataFrame, filter_start_date: datetime, filter_end_date: datetime
) -> pd.DataFrame:
    """
    Filter dataframe to a specific datetime range.

    Args:
        df: pandas dataframe
        filter_start_date: start date to filter data
        filter_end_date: end date to filter data

    Returns:
        df: pandas dataframe
    """
    if "timestamp" not in df.columns:
        raise ValueError("Dataframe must contain 'timestamp' column")

    # Filter by date range
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[
        (df["timestamp"] >= filter_start_date)
        & (df["timestamp"] < filter_end_date + timedelta(days=1))
    ]
    return df


def get_spammers_from_cdr_data(
    cdr_data: pd.DataFrame, threshold_of_calls_per_day: float = 10
) -> list[str]:
    """
    Remove spammers from CDR data based on a threshold of calls per day.
    Args:
        cdr_data: pandas dataframe with CDR data
        threshold_of_calls_per_day: threshold of calls per day to identify spammers
    Returns:
        spammers: list of caller IDs identified as spammers
    """
    # Validate input dataframe
    validate_dataframe(cdr_data, CallDataRecordData)

    # Extract day from timestamp
    cdr_data.loc[:, "day"] = cdr_data["timestamp"].dt.date

    # Get number of transactions per day per transaction type and per caller
    grouped_cdr_data = (
        (cdr_data.groupby(["caller_id", "transaction_type"], as_index=False))
        .apply(
            lambda x: pd.Series(
                {
                    "count_transactions": x.shape[0],
                    "active_days": x["day"].nunique(),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    grouped_cdr_data.loc[:, "avg_calls_per_day"] = (
        grouped_cdr_data["count_transactions"] / grouped_cdr_data["active_days"]
    )

    # Filter out callers with avg calls per day greater than threshold
    spammer_df = grouped_cdr_data.loc[
        grouped_cdr_data["avg_calls_per_day"] > threshold_of_calls_per_day
    ]

    return spammer_df.caller_id.unique().tolist()


def get_outlier_days_from_cdr_data(
    cdr_data: pd.DataFrame, zscore_threshold: float = 2.0
) -> list[str]:
    """
    Remove outlier days from CDR data based on z-score of daily transaction counts.

    Outlier days are those days where the number of transactions is beyond the
    specified z-score threshold from the mean number of daily transactions for each
    transaction type.

    Args:
        cdr_data: pandas dataframe with CDR data
        zscore_threshold: z-score threshold to identify outlier days
    Returns:
        cdr_data: pandas dataframe with outlier days removed
    """
    # Validate input dataframe
    validate_dataframe(cdr_data, CallDataRecordData)

    # Add day column
    cdr_data.loc[:, "day"] = cdr_data["timestamp"].dt.date

    # Group data by caller_id and day to get daily transaction counts
    daily_counts = cdr_data.groupby(["day", "transaction_type"], as_index=False).size()
    daily_counts = daily_counts.rename(columns={"size": "daily_count"})

    # Combine all transaction types to compute overall mean and std
    daily_counts_all_types = daily_counts.groupby("day", as_index=False).agg("sum")

    overall_mean = daily_counts_all_types["daily_count"].mean()
    overall_std = daily_counts_all_types["daily_count"].std()
    overall_bottom_threshold = overall_mean - (zscore_threshold * overall_std)
    overall_top_threshold = overall_mean + (zscore_threshold * overall_std)

    # Get outlier days
    outlier_days = daily_counts_all_types[
        (daily_counts_all_types["daily_count"] < overall_bottom_threshold)
        | (daily_counts_all_types["daily_count"] > overall_top_threshold)
    ]["day"]

    return outlier_days.unique().tolist()


def get_static_diagnostic_statistics(df: pd.DataFrame) -> DataDiagnosticStatistics:
    """
    Get standard diagnostic statistics for CDR, recharge, mobile money and mobile phone data.

    Args:
        df: pandas dataframe
    Returns:
        statistics: DataDiagnosticStatistics object with diagnostic statistics
    """
    if not set(["caller_id", "timestamp"]).issubset(set(df.columns)):
        raise ValueError("Dataframe must contain 'caller_id' and 'timestamp' columns")

    statistics = {
        "total_transactions": int(df.count()["caller_id"]),
        "num_unique_callers": df["caller_id"].nunique(),
        "num_unique_recipients": (
            df["recipient_id"].nunique() if "recipient_id" in df.columns else 0
        ),
        "num_days": (df["timestamp"].max() - df["timestamp"].min()).days + 1,
    }
    return DataDiagnosticStatistics.model_validate(statistics)


def get_timeseries_diagnostic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get timeseries diagnostic statistics for CDR, recharge, mobile money and mobile phone data.

    Args:
        df: pandas dataframe
    Returns:
        statistics: pandas dataframe with timeseries diagnostic statistics

    """
    if not set(["caller_id", "timestamp"]).issubset(set(df.columns)):
        raise ValueError("Dataframe must contain 'caller_id' and 'timestamp' columns")

    df.loc[:, "day"] = df["timestamp"].dt.date
    groupby_columns = ["day"]

    if "transaction_type" in df.columns:
        groupby_columns.append("transaction_type")

    grouped_df = df.groupby(groupby_columns, as_index=False)
    statistics = grouped_df.agg(
        total_transactions=pd.NamedAgg(column="caller_id", aggfunc="count"),
        num_unique_callers=pd.NamedAgg(column="caller_id", aggfunc="nunique"),
        num_unique_recipients=(
            pd.NamedAgg(column="recipient_id", aggfunc="nunique")
            if "recipient_id" in df.columns
            else pd.NamedAgg(column="caller_id", aggfunc=lambda x: 0)
        ),
    )
    return statistics


def identify_daytime(
    spark_df: SparkDataFrame, day_start: int = 7, day_end: int = 19
) -> SparkDataFrame:
    """
    Identify daytime records in the dataframe.

    Args:
        df: Dataframe with a 'timestamp' column
        day_start: Hour to start daytime (inclusive)
        day_end: Hour to end daytime (exclusive)

    Returns:
        df: Dataframe with additional 'is_daytime' column
    """
    if "timestamp" not in spark_df.columns:
        raise ValueError("Dataframe must contain 'timestamp' column")

    spark_df = spark_df.withColumn(
        "is_daytime",
        when(
            (hour(col("timestamp")) >= day_start) & (hour(col("timestamp")) < day_end),
            1,
        ).otherwise(0),
    )
    return spark_df


def identify_weekend(
    spark_df: SparkDataFrame,
    weekend_days: list[int] = [1, 7],
):
    """
    Identify weekend records in the dataframe.

    Args:
        spark_df: Dataframe with a 'timestamp' column
        weekend_days: List of integers representing weekend days (1=Sunday, 7=Saturday)
    Returns:
        df: Dataframe with additional 'is_weekend' column
    """
    if "timestamp" not in spark_df.columns:
        raise ValueError("Dataframe must contain 'timestamp' column")

    spark_df = spark_df.withColumn(
        "is_weekend",
        when((dayofweek(col("day"))).isin(weekend_days), 1).otherwise(0),
    )
    return spark_df


def swap_caller_and_recipient(
    spark_df: SparkDataFrame,
) -> SparkDataFrame:
    """
    Swap caller and recipient columns in the dataframe and append the swapped rows.

    Args:
        spark_df: Dataframe with 'caller_id' and 'recipient_id' columns

    Returns:
        df: Dataframe with swapped caller and recipient columns
    """
    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordData)

    # Add a direction_of_transaction column to indicate incoming/outgoing
    spark_df = spark_df.withColumn(
        "direction_of_transaction", lit(DirectionOfTransactionEnum.OUTGOING.value)
    )

    # Create a copy with swapped caller and recipient columns and
    # direction_of_transaction set to incoming
    spark_df_copy = spark_df.select(
        col("recipient_id").alias("caller_id"),
        col("caller_id").alias("recipient_id"),
        col("caller_antenna_id").alias("recipient_antenna_id"),
        col("recipient_antenna_id").alias("caller_antenna_id"),
        *[
            col(c)
            for c in spark_df.columns
            if c
            not in [
                "caller_id",
                "recipient_id",
                "caller_antenna_id",
                "recipient_antenna_id",
            ]
        ],
    )
    spark_df_copy = spark_df_copy.withColumn(
        "direction_of_transaction", lit(DirectionOfTransactionEnum.INCOMING.value)
    )

    # Append the swapped dataframe to the original dataframe
    spark_df = spark_df.unionByName(spark_df_copy)

    return spark_df


def identify_and_tag_conversations(
    spark_df: SparkDataFrame, max_wait: int = 3600
) -> SparkDataFrame:
    """
    Add conversation ids to interactions in the dataframe.

    From bandicoot's documentation:
    "We define conversations as a series of text messages between the user and one contact.
    A conversation starts with either of the parties sending a text to the other.
    A conversation will stop if no text was exchanged by the parties for an hour or if one of the parties call the other.
    The next conversation will start as soon as a new text is send by either of the parties."
    This functions tags interactions with the conversation id they are part of: the id is the start unix time of the
    conversation.

    Args:
        spark_df: spark dataframe
        max_wait: time (in seconds) after which a conversation ends if no texts or calls have been exchanged

    Returns:
        spark_df: tagged spark dataframe
    """
    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordData)

    window = Window.partitionBy("caller_id", "recipient_id").orderBy("timestamp")

    spark_df = (
        spark_df.withColumn(
            # Cast timestamp to long for time calculations
            "timestamp",
            col("timestamp").cast("long"),
            # Add previous transaction type and timestamp columns
        )
        .withColumn("prev_transaction_type", lag(col("transaction_type")).over(window))
        .withColumn(
            "prev_timestamp",
            lag(col("timestamp")).over(window),
            # Calculate time lapse since previous interaction
        )
        .withColumn(
            "time_lapse",
            col("timestamp") - col("prev_timestamp"),
            # Identify start of new conversations
        )
        .withColumn(
            "conversation",
            when(
                (col("transaction_type") == "text")
                & (
                    (col("prev_transaction_type") == "call")
                    | (col("prev_transaction_type").isNull())
                    | (col("time_lapse") >= max_wait)
                ),
                col("timestamp"),
            ),
            # Identify ongoing conversations
        )
        .withColumn(
            "conversation_last", last("conversation", ignorenulls=True).over(window)
        )
        .withColumn(
            "conversation",
            when(col("conversation").isNotNull(), col("conversation")).otherwise(
                when(col("transaction_type") == "text", col("conversation_last"))
            ),
        )
        # Convert conversation back to timestamp
        .withColumn("conversation", col("conversation").cast("timestamp"))
        # Also convert timestamp back if needed
        .withColumn("timestamp", col("timestamp").cast("timestamp"))
        # Drop intermediate columns
        .drop(
            "prev_transaction_type", "prev_timestamp", "time_lapse", "conversation_last"
        )
    )
    return spark_df


def identify_mobile_money_transaction_direction(
    spark_df: SparkDataFrame,
) -> SparkDataFrame:
    """
    Identify mobile money transaction direction in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id' and 'recipient_id' columns

    Returns:
        df: Dataframe with additional 'direction_of_transaction' column
    """

    # Validate input dataframe
    validate_dataframe(spark_df, MobileMoneyDataWithDay)

    outgoing_interactions = (
        spark_df.select(
            [
                "caller_id",
                "recipient_id",
                "day",
                "amount",
                "caller_balance_before",
                "caller_balance_after",
                "transaction_type",
            ]
        )
        .withColumnRenamed("caller_id", "primary_id")
        .withColumnRenamed("recipient_id", "correspondent_id")
        .withColumnRenamed("caller_balance_before", "balance_before")
        .withColumnRenamed("caller_balance_after", "balance_after")
        .withColumn(
            "direction_of_transaction", lit(DirectionOfTransactionEnum.OUTGOING.value)
        )
    )
    incoming_interactions = (
        spark_df.select(
            [
                "caller_id",
                "recipient_id",
                "day",
                "amount",
                "recipient_balance_before",
                "recipient_balance_after",
                "transaction_type",
            ]
        )
        .withColumnRenamed("recipient_id", "primary_id")
        .withColumnRenamed("caller_id", "correspondent_id")
        .withColumnRenamed("recipient_balance_before", "balance_before")
        .withColumnRenamed("recipient_balance_after", "balance_after")
        .withColumn(
            "direction_of_transaction", lit(DirectionOfTransactionEnum.INCOMING.value)
        )
    )
    all_interactions = outgoing_interactions.unionByName(incoming_interactions)

    return all_interactions
