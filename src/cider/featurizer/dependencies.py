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
from datetime import datetime
from typing import Any
from pyspark.sql import DataFrame as SparkDataFrame

from .schemas import DataDiagnosticStatistics, AllowedPivotColumnsEnum
import pyspark.sql.functions as F
from pyspark.sql.functions import (
    col,
    when,
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
)
from cider.schemas import CallDataRecordTransactionType
import numpy as np


def _get_summary_stats_cols(col_name: str) -> list:
    """
    Get summary statistics columns for a given column name.

    Args:
        col_name: name of the column to get summary statistics for
    """
    return [
        pys_mean(col_name).alias(f"mean_{col_name}"),
        pys_min(col_name).alias(f"min_{col_name}"),
        pys_max(col_name).alias(f"max_{col_name}"),
        stddev_pop(col_name).alias(f"std_{col_name}"),
        expr(f"percentile_approx({col_name}, 0.5)").alias(f"median_{col_name}"),
        skewness(col_name).alias(f"skewness_{col_name}"),
        kurtosis(col_name).alias(f"kurtosis_{col_name}"),
    ]


def _get_agg_columns_by_time_and_transaction_type(
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
        np.meshgrid(*meshgrid_values) if len(meshgrid_values) > 1 else meshgrid_values
    )
    meshgrid = np.array([m.flatten() for m in meshgrid]).T.squeeze()
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
            agg_func(when(condition, col(col_name)).otherwise(0)).alias(agg_name)
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
    # Filter by date range
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[
        (df["timestamp"] >= filter_start_date) & (df["timestamp"] <= filter_end_date)
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
        grouped_cdr_data["avg_calls_per_day"] >= threshold_of_calls_per_day
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
    # Add day column
    cdr_data.loc[:, "day"] = cdr_data["timestamp"].dt.date

    # Group data by caller_id and day to get daily transaction counts
    daily_counts = cdr_data.groupby(["day", "transaction_type"], as_index=False).apply(
        lambda x: x.shape[0],
        include_groups=False,
    )
    daily_counts = daily_counts.rename(columns={None: "daily_count"})

    per_transaction_mean = daily_counts.groupby(
        "transaction_type", as_index=False
    ).daily_count.mean()
    per_transaction_std = daily_counts.groupby(
        "transaction_type", as_index=False
    ).daily_count.std()

    bottom_thresholds = per_transaction_mean.copy()
    top_thresholds = per_transaction_mean.copy()

    bottom_thresholds.daily_count = per_transaction_mean.daily_count - (
        zscore_threshold * per_transaction_std.daily_count
    )
    top_thresholds.daily_count = per_transaction_mean.daily_count + (
        zscore_threshold * per_transaction_std.daily_count
    )

    # Map thresholds to each row's transaction_type
    daily_counts["bottom_threshold"] = daily_counts["transaction_type"].map(
        bottom_thresholds.set_index("transaction_type")["daily_count"]
    )
    daily_counts["top_threshold"] = daily_counts["transaction_type"].map(
        top_thresholds.set_index("transaction_type")["daily_count"]
    )

    # Get outlier days
    outlier_days = daily_counts[
        (daily_counts["daily_count"] < daily_counts["bottom_threshold"])
        | (daily_counts["daily_count"] > daily_counts["top_threshold"])
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
    statistics = {
        "total_transactions": df.shape[0],
        "num_unique_callers": df["caller_id"].nunique(),
        "num_unique_recipients": (
            df["recipient_id"].nunique() if "recipient_id" in df.columns else 0
        ),
        "num_days": df["timestamp"].dt.date.nunique(),
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
