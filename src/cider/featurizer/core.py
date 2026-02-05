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
import numpy as np
from pydantic import BaseModel
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import (
    col,
    count,
    countDistinct,
    lit,
    lag,
    when,
    first,
    mean as pys_mean,
    sum as pys_sum,
    max as pys_max,
    min as pys_min,
    log as pys_log,
    row_number,
    sqrt,
)
from functools import reduce
from datetime import datetime
from pyspark.sql.window import Window
from .schemas import (
    DirectionOfTransactionEnum,
    AllowedPivotColumnsEnum,
    CallDataRecordTagged,
    MobileDataUsageDataWithDay,
    StatsComputationMethodEnum,
    MobileMoneyDataWithDirection,
    RechargeDataWithDay,
    AntennaDataGeometry,
    AntennaDataGeometryWithRegion,
)
from .dependencies import (
    _get_agg_columns_by_cdr_time_and_transaction_type,
    _get_summary_stats_cols,
    _great_circle_distance,
    get_outlier_days_from_cdr_data,
    get_spammers_from_cdr_data,
    filter_to_datetime,
    identify_daytime,
    identify_weekend,
    swap_caller_and_recipient,
    identify_and_tag_conversations,
    identify_mobile_money_transaction_direction,
)
from cider.schemas import (
    AntennaData,
    TransactionScope,
    CallDataRecordData,
    MobileMoneyTransactionData,
    MobileDataUsageData,
    RechargeData,
)
from cider.utils import validate_dataframe, get_spark_session, setup_logger
from pandas import DataFrame as PandasDataFrame


# Setup logging
logger = setup_logger(__name__)


# CDR features
def get_active_days(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Identify active days for each caller in the dataframe, disaggregated by type and time of day.

    Args:
        spark_df: Dataframe with 'caller_id' and 'timestamp' columns

    Returns:
        df: Dataframe with additional 'active_days' column
    """
    logger.info("Calculating active days per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    out = spark_df.groupby("caller_id").agg(
        # Overall
        countDistinct("day").alias("active_days_all"),
        # By weekday/weekend
        countDistinct(when(col("is_weekend") == 0, col("day"))).alias(
            "active_days_weekday"
        ),
        countDistinct(when(col("is_weekend") == 1, col("day"))).alias(
            "active_days_weekend"
        ),
        # By day/night
        countDistinct(when(col("is_daytime") == 1, col("day"))).alias(
            "active_days_day"
        ),
        countDistinct(when(col("is_daytime") == 0, col("day"))).alias(
            "active_days_night"
        ),
        # By both (4 combinations)
        countDistinct(
            when((col("is_weekend") == 0) & (col("is_daytime") == 1), col("day"))
        ).alias("active_days_weekday_day"),
        countDistinct(
            when((col("is_weekend") == 0) & (col("is_daytime") == 0), col("day"))
        ).alias("active_days_weekday_night"),
        countDistinct(
            when((col("is_weekend") == 1) & (col("is_daytime") == 1), col("day"))
        ).alias("active_days_weekend_day"),
        countDistinct(
            when((col("is_weekend") == 1) & (col("is_daytime") == 0), col("day"))
        ).alias("active_days_weekend_night"),
    )

    return out


def get_number_of_contacts_per_caller(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Identify number of unique contacts per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'recipient_id', 'is_weekend', 'is_daytime', and 'transaction_type' columns

    Returns:
        df: Dataframe with num unique callers for each combination of is_weekend, is_daytime, and transaction_type
    """
    logger.info("Calculating number of unique contacts per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    # Count distinct contacts per caller, disaggregated by type and time of day
    spark_df_unique_contacts = spark_df.groupby(
        "caller_id", "is_weekend", "is_daytime", "transaction_type"
    ).agg(countDistinct("recipient_id").alias("num_unique_contacts"))
    aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
        "num_unique_contacts",
        cols_to_use_for_pivot=[
            AllowedPivotColumnsEnum.IS_WEEKEND,
            AllowedPivotColumnsEnum.IS_DAYTIME,
            AllowedPivotColumnsEnum.TRANSACTION_TYPE,
        ],
        agg_func=pys_sum,
    )
    pivoted_df = spark_df_unique_contacts.groupby("caller_id").agg(*aggs)

    # Count distinct contacts per caller, disaggregated by type only
    spark_df_unique_contacts_type_only = spark_df.groupby(
        "caller_id", "transaction_type"
    ).agg(countDistinct("recipient_id").alias("num_unique_contacts"))
    aggs_type_only = _get_agg_columns_by_cdr_time_and_transaction_type(
        "num_unique_contacts",
        cols_to_use_for_pivot=[
            AllowedPivotColumnsEnum.TRANSACTION_TYPE,
        ],
        agg_func=pys_sum,
    )
    pivoted_df_type_only = spark_df_unique_contacts_type_only.groupby("caller_id").agg(
        *aggs_type_only
    )

    # Count distinct contacts per caller, disaggregated by transaction type and weekday/end only
    spark_df_unique_contacts_week_only = spark_df.groupby(
        "caller_id", "is_weekend", "transaction_type"
    ).agg(countDistinct("recipient_id").alias("num_unique_contacts"))
    aggs_time_only = _get_agg_columns_by_cdr_time_and_transaction_type(
        "num_unique_contacts",
        cols_to_use_for_pivot=[
            AllowedPivotColumnsEnum.IS_WEEKEND,
            AllowedPivotColumnsEnum.TRANSACTION_TYPE,
        ],
        agg_func=pys_sum,
    )
    pivoted_df_week_only = spark_df_unique_contacts_week_only.groupby("caller_id").agg(
        *aggs_time_only
    )

    # Count distinct contacts per caller, disaggregated by weekday/end and transaction type only
    spark_df_unique_contacts_day_only = spark_df.groupby(
        "caller_id", "is_daytime", "transaction_type"
    ).agg(countDistinct("recipient_id").alias("num_unique_contacts"))
    aggs_day_only = _get_agg_columns_by_cdr_time_and_transaction_type(
        "num_unique_contacts",
        cols_to_use_for_pivot=[
            AllowedPivotColumnsEnum.IS_DAYTIME,
            AllowedPivotColumnsEnum.TRANSACTION_TYPE,
        ],
        agg_func=pys_sum,
    )
    pivoted_df_day_only = spark_df_unique_contacts_day_only.groupby("caller_id").agg(
        *aggs_day_only
    )

    # Merge all pivoted dataframes
    pivoted_df = reduce(
        lambda df1, df2: df1.join(df2, on="caller_id", how="outer"),
        [pivoted_df, pivoted_df_type_only, pivoted_df_week_only, pivoted_df_day_only],
    )

    return pivoted_df


def get_call_duration_stats(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Get call duration statistics per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'is_weekend', 'is_daytime', and 'transaction_type' columns

    Returns:
        df: Dataframe with call duration statistics columns for each weekday/weekend and day/nighttime combination.
    """
    logger.info("Calculating call duration statistics per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    filtered_df = spark_df.filter(col("transaction_type") == "call")

    summary_stats_cols = _get_summary_stats_cols("duration")
    stats_df = filtered_df.groupby(
        "caller_id", "is_weekend", "is_daytime", "transaction_type"
    ).agg(*summary_stats_cols)

    all_stats_aggs = []
    for stats_col in [e.value for e in StatsComputationMethodEnum]:
        aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
            f"{stats_col}_duration",
            cols_to_use_for_pivot=[
                AllowedPivotColumnsEnum.IS_WEEKEND,
                AllowedPivotColumnsEnum.IS_DAYTIME,
            ],
            agg_func=first,
        )
        all_stats_aggs.extend(aggs)

    pivoted_df = stats_df.groupby("caller_id").agg(*all_stats_aggs)

    return pivoted_df


def get_percentage_of_nocturnal_interactions(
    spark_df: SparkDataFrame,
) -> SparkDataFrame:
    """
    Get percentage of nocturnal interactions per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'is_daytime', 'transaction_type' columns

    Returns:
        df: Dataframe with percentage of nocturnal interactions column
    """
    logger.info("Calculating percentage of nocturnal interactions per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    count_df = (
        spark_df.withColumn(
            "is_nocturnal", when(col("is_daytime") == 0, 1).otherwise(0)
        )
        .groupby("caller_id", "is_weekend", "transaction_type")
        .agg(pys_mean("is_nocturnal").alias("percentage_nocturnal_interactions"))
    )

    aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
        "percentage_nocturnal_interactions",
        cols_to_use_for_pivot=[
            AllowedPivotColumnsEnum.IS_WEEKEND,
            AllowedPivotColumnsEnum.TRANSACTION_TYPE,
        ],
        agg_func=pys_sum,
    )
    pivoted_df = count_df.groupby("caller_id").agg(*aggs)

    count_df_all = (
        spark_df.withColumn(
            "is_nocturnal", when(col("is_daytime") == 0, 1).otherwise(0)
        )
        .groupby("caller_id", "transaction_type")
        .agg(pys_mean("is_nocturnal").alias("percentage_nocturnal_interactions"))
    )
    aggs_all_week = _get_agg_columns_by_cdr_time_and_transaction_type(
        "percentage_nocturnal_interactions",
        cols_to_use_for_pivot=[
            AllowedPivotColumnsEnum.TRANSACTION_TYPE,
        ],
        agg_func=pys_sum,
    )
    pivoted_df_all_week = (
        count_df_all.groupby("caller_id").agg(*aggs_all_week).drop("is_weekend")
    )
    return pivoted_df.join(pivoted_df_all_week, on="caller_id", how="inner")


def get_percentage_of_initiated_conversations(
    spark_df: SparkDataFrame,
) -> SparkDataFrame:
    """
    Get percentage of initiated conversations per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'timestamp', 'conversation', 'is_weekend', 'direction_of_transaction' and 'is_daytime' columns

    Returns:
        df: Dataframe with percentage of initiated conversations column
    """
    logger.info("Calculating percentage of initiated conversations per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    convo_df = spark_df.where(
        (col("conversation").isNotNull()) & (col("transaction_type") == "text")
    ).withColumn(
        "initiated_conversation",
        when(col("direction_of_transaction") == "outgoing", 1).otherwise(0),
    )

    def _get_groupby_and_pivot_df(
        groupby_cols: list[str],
        pivot_cols: list[AllowedPivotColumnsEnum],
        drop_cols: list[str] = [],
    ) -> SparkDataFrame:
        convo_df_grouped = convo_df.groupby(*groupby_cols).agg(
            pys_mean("initiated_conversation").alias(
                "percentage_initiated_conversations"
            )
        )
        aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
            "percentage_initiated_conversations",
            cols_to_use_for_pivot=pivot_cols,
            agg_func=pys_sum,
        )
        convo_df_grouped = convo_df_grouped.groupby("caller_id").agg(*aggs)
        if drop_cols:
            convo_df_grouped = convo_df_grouped.drop(*drop_cols)
        return convo_df_grouped

    pivot_df = _get_groupby_and_pivot_df(
        ["caller_id", "is_weekend", "is_daytime"],
        [
            AllowedPivotColumnsEnum.IS_WEEKEND,
            AllowedPivotColumnsEnum.IS_DAYTIME,
        ],
    )
    pivot_df_all_week = _get_groupby_and_pivot_df(
        ["caller_id", "is_daytime"],
        [
            AllowedPivotColumnsEnum.IS_DAYTIME,
        ],
    )
    pivot_df_all_day = _get_groupby_and_pivot_df(
        ["caller_id", "is_weekend"],
        [
            AllowedPivotColumnsEnum.IS_WEEKEND,
        ],
    )

    return (
        pivot_df.join(pivot_df_all_week, on="caller_id", how="inner")
        .join(pivot_df_all_day, on="caller_id", how="inner")
        .join(
            convo_df.groupby("caller_id").agg(
                pys_mean("initiated_conversation").alias(
                    "percentage_initiated_conversations"
                )
            ),
            on="caller_id",
            how="inner",
        )
    )


def get_percentage_of_initiated_calls(
    spark_df: SparkDataFrame,
) -> SparkDataFrame:
    """
    Get percentage of initiated calls per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'is_weekend', 'is_daytime', 'direction_of_transaction' and 'transaction_type' columns

    Returns:
        df: Dataframe with percentage of initiated calls column
    """
    logger.info("Calculating percentage of initiated calls per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    spark_df_filtered = spark_df.where(col("transaction_type") == "call")

    filtered_df = spark_df_filtered.withColumn(
        "initiated_call",
        when(col("direction_of_transaction") == "outgoing", 1).otherwise(0),
    )

    def _get_groupby_and_pivot_df(
        groupby_cols: list[str],
        pivot_cols: list[AllowedPivotColumnsEnum],
        drop_cols: list[str] = [],
    ) -> SparkDataFrame:
        interaction_df = filtered_df.groupby(*groupby_cols).agg(
            pys_mean("initiated_call").alias("percentage_initiated_calls")
        )
        aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
            "percentage_initiated_calls",
            cols_to_use_for_pivot=pivot_cols,
            agg_func=pys_sum,
        )
        interaction_df = interaction_df.groupby("caller_id").agg(*aggs)
        if drop_cols:
            interaction_df = interaction_df.drop(*drop_cols)
        return interaction_df

    interaction_df = _get_groupby_and_pivot_df(
        ["caller_id", "is_weekend", "is_daytime"],
        [AllowedPivotColumnsEnum.IS_WEEKEND, AllowedPivotColumnsEnum.IS_DAYTIME],
    )
    interaction_df_all_week = _get_groupby_and_pivot_df(
        ["caller_id", "is_daytime"],
        [AllowedPivotColumnsEnum.IS_DAYTIME],
        ["is_weekend"],
    )
    interaction_df_all_day = _get_groupby_and_pivot_df(
        ["caller_id", "is_weekend"],
        [AllowedPivotColumnsEnum.IS_WEEKEND],
        ["is_daytime"],
    )

    return (
        interaction_df.join(interaction_df_all_week, on="caller_id", how="inner")
        .join(interaction_df_all_day, on="caller_id", how="inner")
        .join(
            filtered_df.groupby("caller_id").agg(
                pys_mean("initiated_call").alias("percentage_initiated_calls")
            ),
            on="caller_id",
            how="inner",
        )
    )


def get_text_response_time_delay_stats(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Get text response time delay statistics per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'recipient_id', 'transaction_type', 'timestamp', 'is_weekend', 'is_daytime', 'conversation' and 'direction_of_transaction' columns

    Returns:
        df: Dataframe with text response time delay statistics columns
    """
    logger.info("Calculating text response time delay statistics per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    # Filter to only text transactions
    filtered_df = spark_df.filter(col("transaction_type") == "text")

    window = Window.partitionBy("caller_id", "recipient_id", "conversation").orderBy(
        "timestamp"
    )

    # Calculate time difference between consecutive texts
    summary_stats_cols = _get_summary_stats_cols("response_time_delay")
    response_time_df = (
        filtered_df.withColumn(
            "prev_direction", lag(col("direction_of_transaction")).over(window)
        )
        .withColumn("prev_timestamp", lag(col("timestamp")).over(window))
        .withColumn(
            "response_time_delay",
            when(
                (col("direction_of_transaction") == "outgoing")
                & (col("prev_direction") == "incoming"),
                col("timestamp").cast("long") - col("prev_timestamp").cast("long"),
            ),
        )
        .groupby("caller_id", "is_weekend", "is_daytime")
        .agg(*summary_stats_cols)
    )

    all_aggs = []
    for pivot_col in [e.value for e in StatsComputationMethodEnum]:
        aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
            f"{pivot_col}_response_time_delay",
            cols_to_use_for_pivot=[
                AllowedPivotColumnsEnum.IS_WEEKEND,
                AllowedPivotColumnsEnum.IS_DAYTIME,
            ],
            agg_func=first,
        )
        all_aggs.extend(aggs)

    stats_df = response_time_df.groupby("caller_id").agg(*all_aggs)

    return stats_df


def get_text_response_rate(
    spark_df: SparkDataFrame,
) -> SparkDataFrame:
    """
    Get text response rate per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'recipient_id', 'transaction_type', 'timestamp', 'is_weekend', 'is_daytime', 'conversation' and 'direction_of_transaction' columns

    Returns:
        df: Dataframe with text response rate columns
    """
    logger.info("Calculating text response rate per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    # Filter to only text transactions
    filtered_df = spark_df.filter(col("transaction_type") == "text")

    window = Window.partitionBy("caller_id", "recipient_id", "conversation")

    # Calculate response rate
    response_df = (
        filtered_df.withColumn(
            "direction",
            when((col("direction_of_transaction") == "outgoing"), 1).otherwise(0),
        )
        .withColumn("responded", pys_max(col("direction")).over(window))
        .where(
            (col("conversation") == col("timestamp"))
            & (col("direction_of_transaction") == "incoming")
        )
    )

    def _get_groupby_and_pivot_df(
        groupby_cols: list[str],
        pivot_cols: list[AllowedPivotColumnsEnum],
        drop_cols: list[str] = [],
    ) -> SparkDataFrame:
        df = response_df.groupby(*groupby_cols).agg(
            pys_mean("responded").alias("text_response_rate")
        )
        aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
            "text_response_rate",
            cols_to_use_for_pivot=pivot_cols,
            agg_func=pys_sum,
        )
        out_df = df.groupby("caller_id").agg(*aggs)
        if drop_cols:
            out_df = out_df.drop(*drop_cols)
        return out_df

    response_rate_df = _get_groupby_and_pivot_df(
        ["caller_id", "is_weekend", "is_daytime"],
        [
            AllowedPivotColumnsEnum.IS_WEEKEND,
            AllowedPivotColumnsEnum.IS_DAYTIME,
        ],
    )

    response_rate_df_all_week = _get_groupby_and_pivot_df(
        ["caller_id", "is_daytime"],
        [AllowedPivotColumnsEnum.IS_DAYTIME],
        ["is_weekend"],
    )
    response_rate_df_all_day = _get_groupby_and_pivot_df(
        ["caller_id", "is_weekend"],
        [AllowedPivotColumnsEnum.IS_WEEKEND],
        ["is_daytime"],
    )

    return (
        response_rate_df.join(response_rate_df_all_week, on="caller_id", how="inner")
        .join(response_rate_df_all_day, on="caller_id", how="inner")
        .join(
            response_df.groupby("caller_id").agg(
                pys_mean("responded").alias("text_response_rate")
            ),
            on="caller_id",
            how="inner",
        )
    )


def get_entropy_of_interactions_per_caller(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Get entropy of interactions per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'recipient_id', 'is_weekend', 'is_daytime', 'transaction_type' columns

    Returns:
        df: Dataframe with entropy of interactions column
    """
    logger.info("Calculating entropy of interactions per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    def _get_groupby_and_pivot_df(
        groupby_cols: list[str],
        pivot_cols: list[AllowedPivotColumnsEnum],
        drop_cols: list[str] = [],
    ) -> SparkDataFrame:
        window = Window.partitionBy(groupby_cols)
        entropy_df = (
            spark_df.groupby("recipient_id", *groupby_cols)
            .agg(count(lit(0)).alias("interaction_count"))
            .withColumn("total_count", pys_sum("interaction_count").over(window))
            .withColumn(
                "probability",
                (col("interaction_count") / col("total_count").cast("float")),
            )
            .groupby(groupby_cols)
            .agg(
                (-1 * pys_sum(col("probability") * pys_log(col("probability")))).alias(
                    "entropy_of_interactions"
                )
            )
        )
        aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
            "entropy_of_interactions",
            cols_to_use_for_pivot=pivot_cols,
            agg_func=pys_sum,
        )
        out_df = entropy_df.groupby("caller_id").agg(*aggs)
        if drop_cols:
            out_df = out_df.drop(*drop_cols)
        return out_df

    entropy_df = _get_groupby_and_pivot_df(
        ["caller_id", "is_weekend", "is_daytime", "transaction_type"],
        [
            AllowedPivotColumnsEnum.IS_WEEKEND,
            AllowedPivotColumnsEnum.IS_DAYTIME,
            AllowedPivotColumnsEnum.TRANSACTION_TYPE,
        ],
    )
    entropy_df_all_week = _get_groupby_and_pivot_df(
        ["caller_id", "is_daytime", "transaction_type"],
        [
            AllowedPivotColumnsEnum.IS_DAYTIME,
            AllowedPivotColumnsEnum.TRANSACTION_TYPE,
        ],
        # ["is_weekend"],
    )
    entropy_df_all_day = _get_groupby_and_pivot_df(
        ["caller_id", "is_weekend", "transaction_type"],
        [
            AllowedPivotColumnsEnum.IS_WEEKEND,
            AllowedPivotColumnsEnum.TRANSACTION_TYPE,
        ],
        # ["is_daytime"],
    )
    entropy_df_all = _get_groupby_and_pivot_df(
        ["caller_id", "transaction_type"],
        [AllowedPivotColumnsEnum.TRANSACTION_TYPE],
        # ["is_weekend", "is_daytime"],
    )

    return (
        entropy_df.join(entropy_df_all_week, on="caller_id", how="inner")
        .join(entropy_df_all_day, on="caller_id", how="inner")
        .join(entropy_df_all, on="caller_id", how="inner")
    )


def get_outgoing_interaction_fraction_stats(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Get outgoing call fraction statistics per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'recipient_id', 'is_weekend', 'is_daytime', 'direction_of_transaction', 'transaction_type' columns

    Returns:
        df: Dataframe with outgoing call fraction statistics columns
    """
    logger.info("Calculating outgoing interaction fraction statistics per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    # Get interaction counts per caller-recipient pair
    count_df = (
        spark_df.groupby(
            "caller_id",
            "recipient_id",
            "is_weekend",
            "is_daytime",
            "transaction_type",
            "direction_of_transaction",
        )
        .agg(count(lit(0)).alias("interaction_count"))
        .groupby(
            "caller_id", "recipient_id", "is_weekend", "is_daytime", "transaction_type"
        )
        .pivot("direction_of_transaction")
        .agg(first("interaction_count").alias("interaction_count"))
        .fillna(0)
    )

    # Add columns for missing direction of transactions
    missing_direction_cols = [
        e.value for e in DirectionOfTransactionEnum if e.value not in count_df.columns
    ]
    for col_name in missing_direction_cols:
        count_df = count_df.withColumn(col_name, lit(0))

    # Calculate outgoing call fraction and corresponding stats
    summary_stats_cols = _get_summary_stats_cols("fraction_of_outgoing_calls")
    fraction_df = (
        count_df.withColumn(
            "total_interactions",
            sum([col(e.value) for e in DirectionOfTransactionEnum]),
        )
        .withColumn(
            "fraction_of_outgoing_calls",
            col(DirectionOfTransactionEnum.OUTGOING.value)
            / col("total_interactions").cast("float"),
        )
        .groupby("caller_id", "is_weekend", "is_daytime", "transaction_type")
        .agg(*summary_stats_cols)
    )

    all_aggs = []
    cols_to_pivot = [
        f"{e.value}_fraction_of_outgoing_calls" for e in StatsComputationMethodEnum
    ]

    for pivot_col in cols_to_pivot:
        aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
            pivot_col,
            cols_to_use_for_pivot=[
                AllowedPivotColumnsEnum.IS_WEEKEND,
                AllowedPivotColumnsEnum.IS_DAYTIME,
                AllowedPivotColumnsEnum.TRANSACTION_TYPE,
            ],
            agg_func=first,
        )
        all_aggs.extend(aggs)

    stats_df = fraction_df.groupby("caller_id").agg(*all_aggs)

    return stats_df


def get_interaction_stats_per_caller(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Get interaction statistics per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'recipient_id', 'is_weekend', 'is_daytime', 'transaction_type' columns

    Returns:
        df: Dataframe with interaction statistics columns
    """
    logger.info("Calculating interaction statistics per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    summary_stats_cols = _get_summary_stats_cols("interaction_count")
    interaction_df = (
        spark_df.groupby(
            "caller_id", "recipient_id", "is_weekend", "is_daytime", "transaction_type"
        )
        .agg(count(lit(0)).alias("interaction_count"))
        .groupby("caller_id", "is_weekend", "is_daytime", "transaction_type")
        .agg(*summary_stats_cols)
    )

    all_aggs = []
    cols_to_pivot = [f"{e.value}_interaction_count" for e in StatsComputationMethodEnum]
    for pivot_col in cols_to_pivot:
        aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
            pivot_col,
            cols_to_use_for_pivot=[
                AllowedPivotColumnsEnum.IS_WEEKEND,
                AllowedPivotColumnsEnum.IS_DAYTIME,
                AllowedPivotColumnsEnum.TRANSACTION_TYPE,
            ],
            agg_func=first,
        )
        all_aggs.extend(aggs)

    pivoted_df = interaction_df.groupby("caller_id").agg(*all_aggs)

    return pivoted_df


def get_inter_event_time_stats(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Get inter-event time statistics per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'timestamp', 'is_weekend', 'is_daytime', 'transaction_type' columns

    Returns:
        df: Dataframe with inter-event time statistics columns
    """
    logger.info("Calculating inter-event time statistics per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    # Calculate inter-event times and corresponding summary stats
    window = Window.partitionBy(
        "caller_id", "is_weekend", "is_daytime", "transaction_type"
    ).orderBy("timestamp")

    summary_stats_cols = _get_summary_stats_cols("inter_event_time")
    inter_event_df = (
        spark_df.withColumn("timestamp_long", col("timestamp").cast("long"))
        .withColumn("prev_timestamp", lag(col("timestamp_long")).over(window))
        .withColumn("inter_event_time", col("timestamp_long") - col("prev_timestamp"))
        .groupby("caller_id", "is_weekend", "is_daytime", "transaction_type")
        .agg(*summary_stats_cols)
    )

    # Pivot inter-event time stats
    all_aggs = []
    cols_to_pivot = [f"{e.value}_inter_event_time" for e in StatsComputationMethodEnum]
    for pivot_col in cols_to_pivot:
        aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
            pivot_col,
            cols_to_use_for_pivot=[
                AllowedPivotColumnsEnum.IS_WEEKEND,
                AllowedPivotColumnsEnum.IS_DAYTIME,
                AllowedPivotColumnsEnum.TRANSACTION_TYPE,
            ],
            agg_func=first,
        )
        all_aggs.extend(aggs)

    pivoted_df = inter_event_df.groupby("caller_id").agg(*all_aggs)
    return pivoted_df


def get_pareto_principle_interaction_stats(
    spark_df: SparkDataFrame,
    percentage_threshold: float = 0.8,
) -> SparkDataFrame:
    """
    The Pareto principle (80/20 rule) states that roughly 80% of effects come from 20% of causes.
    This function calculates the fraction of recipients that account for `threshold` percentage of
    a caller's interactions, disaggregated by weekday/weekend and daytime/nighttime interactions.

    Args:
        spark_df: Dataframe with 'caller_id', 'recipient_id', 'is_weekend', 'is_daytime', 'transaction_type' columns
        percentage_threshold: The percentage threshold to calculate the Pareto principle for (default is 0.8 for 80%)

    Returns:
        df: Dataframe with Pareto principle interaction statistics columns
    """
    logger.info("Calculating Pareto principle interaction statistics per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    # Set up windows for calculations
    window_1 = Window.partitionBy(
        "caller_id", "is_weekend", "is_daytime", "transaction_type"
    )
    window_2 = Window.partitionBy(
        "caller_id", "is_weekend", "is_daytime", "transaction_type"
    ).orderBy(col("interaction_count").desc())
    window_3 = Window.partitionBy(
        "caller_id", "is_weekend", "is_daytime", "transaction_type"
    ).orderBy("row_number")

    # Calculate Pareto principle interaction stats
    pareto_interaction_df = (
        spark_df.groupby(
            "caller_id", "recipient_id", "is_weekend", "is_daytime", "transaction_type"
        )
        .agg(count(lit(0)).alias("interaction_count"))
        .withColumn("total_interactions", pys_sum("interaction_count").over(window_1))
        .withColumn("row_number", row_number().over(window_2))
        .withColumn(
            "cumulative_interactions", pys_sum("interaction_count").over(window_3)
        )
        .withColumn(
            "cumulative_interaction_fraction",
            col("cumulative_interactions") / col("total_interactions").cast("float"),
        )
        .withColumn(
            "row_number",
            when(
                col("cumulative_interaction_fraction") >= percentage_threshold,
                col("row_number"),
            ),
        )
        .groupby("caller_id", "is_weekend", "is_daytime", "transaction_type")
        .agg(
            pys_min("row_number").alias("num_pareto_callers"),
            countDistinct("recipient_id").alias("num_unique_recipients"),
        )
        .withColumn(
            "pareto_principle_interaction_fraction",
            col("num_pareto_callers") / col("num_unique_recipients").cast("float"),
        )
    )

    aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
        "pareto_principle_interaction_fraction",
        cols_to_use_for_pivot=[
            AllowedPivotColumnsEnum.IS_WEEKEND,
            AllowedPivotColumnsEnum.IS_DAYTIME,
            AllowedPivotColumnsEnum.TRANSACTION_TYPE,
        ],
        agg_func=first,
    )
    pivoted_df = pareto_interaction_df.groupby("caller_id").agg(*aggs)
    return pivoted_df


def get_pareto_principle_call_duration_stats(
    spark_df: SparkDataFrame,
    percentage_threshold: float = 0.8,
) -> SparkDataFrame:
    """
    The Pareto principle (80/20 rule) states that roughly 80% of effects come from 20% of causes.
    This function calculates the fraction of recipients that account for `threshold` percentage of
    a caller's call duration, disaggregated by weekday/weekend and daytime/nighttime interactions.

    Args:
        spark_df: Dataframe with 'caller_id', 'recipient_id', 'is_weekend', 'is_daytime', 'transaction_type', 'duration' columns
        percentage_threshold: The percentage threshold to calculate the Pareto principle for (default is 0.8 for 80%)

    Returns:
        df: Dataframe with Pareto principle call duration statistics columns
    """
    logger.info("Calculating Pareto principle call duration statistics per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    # Filter to only call transactions
    filtered_df = spark_df.filter(col("transaction_type") == "call")

    # Set up windows for calculations
    window_1 = Window.partitionBy("caller_id", "is_weekend", "is_daytime")
    window_2 = Window.partitionBy("caller_id", "is_weekend", "is_daytime").orderBy(
        col("total_call_duration").desc()
    )
    window_3 = Window.partitionBy("caller_id", "is_weekend", "is_daytime").orderBy(
        "row_number"
    )

    # Calculate Pareto principle call duration stats
    pareto_call_duration_df = (
        filtered_df.groupby("caller_id", "recipient_id", "is_weekend", "is_daytime")
        .agg(pys_sum("duration").alias("total_call_duration"))
        .withColumn(
            "overall_call_duration", pys_sum("total_call_duration").over(window_1)
        )
        .withColumn("row_number", row_number().over(window_2))
        .withColumn(
            "cumulative_call_duration", pys_sum("total_call_duration").over(window_3)
        )
        .withColumn(
            "cumulative_call_fraction",
            col("cumulative_call_duration")
            / col("overall_call_duration").cast("float"),
        )
        .withColumn(
            "row_number",
            when(
                col("cumulative_call_fraction") >= percentage_threshold,
                col("row_number"),
            ),
        )
        .groupby("caller_id", "is_weekend", "is_daytime")
        .agg(
            pys_min("row_number").alias("num_pareto_callers"),
            countDistinct("recipient_id").alias("num_unique_recipients"),
        )
        .withColumn(
            "pareto_call_duration_fraction",
            col("num_pareto_callers") / col("num_unique_recipients").cast("float"),
        )
    )

    aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
        "pareto_call_duration_fraction",
        cols_to_use_for_pivot=[
            AllowedPivotColumnsEnum.IS_WEEKEND,
            AllowedPivotColumnsEnum.IS_DAYTIME,
        ],
        agg_func=first,
    )
    pivoted_df = pareto_call_duration_df.groupby("caller_id").agg(*aggs)
    return pivoted_df


def get_number_of_interactions_per_user(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Get number of interactions per user in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'is_weekend', 'is_daytime', 'transaction_type', 'direction_of_transaction' columns

    Returns:
        df: Dataframe with number of interactions columns
    """
    logger.info("Calculating number of interactions per user")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    def _get_groupby_and_pivot_df(
        groupby_cols: list[str],
        pivot_cols: list[AllowedPivotColumnsEnum],
        drop_cols: list[str] = [],
    ):

        count_df = spark_df.groupby(
            groupby_cols,
        ).agg(count(lit(0)).alias("num_interactions"))

        if "direction_of_transaction" in groupby_cols:
            # Pivot by direction first
            pivoted_df = (
                count_df.groupby(
                    [col for col in groupby_cols if col != "direction_of_transaction"]
                )
                .pivot(
                    "direction_of_transaction",
                    [e.value for e in DirectionOfTransactionEnum],
                )
                .agg(first("num_interactions"))
            )

            # Rename direction columns and aggregate
            all_aggs = []
            for e in DirectionOfTransactionEnum:
                pivoted_df = pivoted_df.withColumnRenamed(
                    e.value, f"{e.value}_num_interactions"
                )
                aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
                    f"{e.value}_num_interactions",
                    cols_to_use_for_pivot=pivot_cols,
                    agg_func=pys_sum,
                )
                all_aggs.extend(aggs)

            # Final aggregation by caller_id only
            pivoted_df = pivoted_df.groupby("caller_id").agg(*all_aggs)
        else:
            aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
                "num_interactions",
                cols_to_use_for_pivot=pivot_cols,
                agg_func=pys_sum,
            )
            pivoted_df = count_df.groupby("caller_id").agg(*aggs)

        if drop_cols:
            pivoted_df = pivoted_df.drop(*drop_cols)

        return pivoted_df

    base_cols = ["caller_id", "transaction_type"]
    base_pivots = [AllowedPivotColumnsEnum.TRANSACTION_TYPE]
    dimensions = {
        "is_weekend": [True, False],
        "is_daytime": [True, False],
        "direction_of_transaction": [True, False],
    }
    drop_cols = {
        "is_weekend": ["is_daytime"],
        "is_daytime": ["is_weekend"],
        "direction_of_transaction": [],
    }
    pivot_cols = {
        "is_weekend": [AllowedPivotColumnsEnum.IS_WEEKEND],
        "is_daytime": [AllowedPivotColumnsEnum.IS_DAYTIME],
        "direction_of_transaction": [],
    }
    meshgrid_for_dimensions = (
        np.array(np.meshgrid(*dimensions.values())).reshape(len(dimensions), -1).T
    )
    dfs_to_join = []
    for setting in meshgrid_for_dimensions:
        groupby_cols = base_cols.copy()
        cols_to_use_for_pivot = base_pivots.copy()
        cols_to_drop = []
        for i, dim in enumerate(dimensions.keys()):
            if setting[i]:
                groupby_cols.append(dim)
                cols_to_use_for_pivot.extend(pivot_cols[dim])
                if drop_cols[dim]:
                    cols_to_drop.extend(drop_cols[dim])

        df = _get_groupby_and_pivot_df(
            groupby_cols,
            cols_to_use_for_pivot,
            cols_to_drop,
        )
        dfs_to_join.append(df)
    return reduce(
        lambda left, right: left.join(right, on="caller_id", how="inner"), dfs_to_join
    )


def get_number_of_antennas(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Get number of unique antennas per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'caller_antenna_id', 'is_daytime', 'is_weekend' columns

    Returns:
        df: Dataframe with number of unique antennas column
    """
    logger.info("Calculating number of unique antennas per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    def _get_groupby_and_pivot_df(
        groupby_cols: list[str],
        pivot_cols: list[AllowedPivotColumnsEnum],
        drop_cols: list[str] = [],
    ):
        antenna_df = spark_df.groupby(groupby_cols).agg(
            countDistinct("caller_antenna_id").alias("num_unique_antennas")
        )

        aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
            "num_unique_antennas",
            cols_to_use_for_pivot=pivot_cols,
            agg_func=pys_sum,
        )
        pivoted_df = antenna_df.groupby("caller_id").agg(*aggs)

        if drop_cols:
            pivoted_df = pivoted_df.drop(*drop_cols)

        return pivoted_df

    base_cols = ["caller_id"]
    base_pivots: list[AllowedPivotColumnsEnum] = []
    dimensions = {
        "is_weekend": [True, False],
        "is_daytime": [True, False],
    }
    drop_cols = {
        "is_weekend": ["is_daytime"],
        "is_daytime": ["is_weekend"],
    }
    pivot_cols = {
        "is_weekend": [AllowedPivotColumnsEnum.IS_WEEKEND],
        "is_daytime": [AllowedPivotColumnsEnum.IS_DAYTIME],
    }
    meshgrid_for_dimensions = (
        np.array(np.meshgrid(*dimensions.values())).reshape(len(dimensions), -1).T
    )
    dfs_to_join = []
    for setting in meshgrid_for_dimensions:
        groupby_cols = base_cols.copy()
        cols_to_use_for_pivot = base_pivots.copy()
        cols_to_drop = []
        if setting.sum() == 0:
            # If no dimensions are selected, we want to group by caller_id only and not pivot
            df = spark_df.groupby("caller_id").agg(
                countDistinct("caller_antenna_id").alias("num_unique_antennas")
            )
        else:
            for i, dim in enumerate(dimensions.keys()):
                if setting[i]:
                    groupby_cols.append(dim)
                    cols_to_use_for_pivot.extend(pivot_cols[dim])
                    if drop_cols[dim]:
                        cols_to_drop.extend(drop_cols[dim])

            df = _get_groupby_and_pivot_df(
                groupby_cols,
                cols_to_use_for_pivot,
                cols_to_drop,
            )
        dfs_to_join.append(df)
    return reduce(
        lambda left, right: left.join(right, on="caller_id", how="inner"), dfs_to_join
    )


def get_entropy_of_antennas_per_caller(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Get entropy of antennas per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'caller_antenna_id', 'is_daytime', 'is_weekend' columns

    Returns:
        df: Dataframe with entropy of antennas column
    """
    logger.info("Calculating entropy of antennas per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    def _get_groupby_and_pivot_df(
        groupby_cols: list[str],
        pivot_cols: list[AllowedPivotColumnsEnum],
        drop_cols: list[str] = [],
    ):
        # Build window based on groupby_cols
        window = Window.partitionBy(*groupby_cols)

        entropy_df = (
            spark_df.groupby(groupby_cols + ["caller_antenna_id"])
            .agg(count(lit(0)).alias("interaction_count"))
            .withColumn("total_count", pys_sum("interaction_count").over(window))
            .withColumn(
                "fraction_of_interactions",
                (col("interaction_count") / col("total_count").cast("float")),
            )
            .groupby(groupby_cols)
            .agg(
                (
                    -1
                    * pys_sum(
                        col("fraction_of_interactions")
                        * pys_log(col("fraction_of_interactions"))
                    )
                ).alias("entropy_of_antennas")
            )
        )

        if pivot_cols:
            aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
                "entropy_of_antennas",
                cols_to_use_for_pivot=pivot_cols,
                agg_func=pys_sum,
            )
            pivoted_df = entropy_df.groupby("caller_id").agg(*aggs)
        else:
            # No pivoting needed, just rename the column
            pivoted_df = entropy_df.select("caller_id", "entropy_of_antennas")

        if drop_cols:
            pivoted_df = pivoted_df.drop(*drop_cols)

        return pivoted_df

    base_cols = ["caller_id"]
    base_pivots: list[AllowedPivotColumnsEnum] = []
    dimensions = {
        "is_weekend": [True, False],
        "is_daytime": [True, False],
    }
    drop_cols = {
        "is_weekend": ["is_daytime"],
        "is_daytime": ["is_weekend"],
    }
    pivot_cols = {
        "is_weekend": [AllowedPivotColumnsEnum.IS_WEEKEND],
        "is_daytime": [AllowedPivotColumnsEnum.IS_DAYTIME],
    }
    meshgrid_for_dimensions = (
        np.array(np.meshgrid(*dimensions.values())).reshape(len(dimensions), -1).T
    )
    dfs_to_join = []
    for setting in meshgrid_for_dimensions:
        groupby_cols = base_cols.copy()
        cols_to_use_for_pivot = base_pivots.copy()
        cols_to_drop = []
        if setting.sum() == 0:
            # No dimensions case
            df = _get_groupby_and_pivot_df(groupby_cols, cols_to_use_for_pivot)
        else:
            for i, (dim_name, dim_value) in enumerate(dimensions.items()):
                if setting[i]:
                    groupby_cols.append(dim_name)
                    cols_to_use_for_pivot.extend(pivot_cols[dim_name])
                else:
                    cols_to_drop.extend(drop_cols[dim_name])
            df = _get_groupby_and_pivot_df(
                groupby_cols, cols_to_use_for_pivot, cols_to_drop
            )
        dfs_to_join.append(df)
    return reduce(
        lambda left, right: left.join(right, on="caller_id", how="inner"), dfs_to_join
    )


def get_radius_of_gyration(
    spark_df: SparkDataFrame, spark_antennas_df: SparkDataFrame
) -> SparkDataFrame:
    """
    Returns the radius of gyration of users, disaggregated by type and time of day

    References
    ----------
    .. [GON2008] Gonzalez, M. C., Hidalgo, C. A., & Barabasi, A. L. (2008).
        Understanding individual human mobility patterns. Nature, 453(7196),
        779-782.

    Args:
        spark_df: Dataframe with 'caller_id', 'caller_antenna_id', 'is_weekend', 'is_daytime' columns
        spark_antennas_df: Dataframe with 'caller_antenna_id', 'latitude', 'longitude' columns

    Returns:
        df: Dataframe with radius of gyration column
    """
    logger.info("Calculating radius of gyration per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)
    validate_dataframe(spark_antennas_df, AntennaDataGeometry)

    # Join antennas and CDR data
    joined_df = spark_df.join(spark_antennas_df, on="caller_antenna_id", how="inner")

    def _get_groupby_and_pivot_df(
        groupby_cols: list[str],
        pivot_cols: list[AllowedPivotColumnsEnum],
        drop_cols: list[str] = [],
    ) -> SparkDataFrame:
        # Calculate center of mass coordinates
        coordinates_df = (
            joined_df.groupby(groupby_cols)
            .agg(
                pys_sum("latitude").alias("sum_latitude"),
                pys_sum("longitude").alias("sum_longitude"),
                count(lit(0)).alias("num_records"),
            )
            .withColumn(
                "center_of_mass_latitude",
                col("sum_latitude") / col("num_records").cast("float"),
            )
            .withColumn(
                "center_of_mass_longitude",
                col("sum_longitude") / col("num_records").cast("float"),
            )
            .drop("latitude", "longitude")
        )

        coordinates_df = joined_df.join(
            coordinates_df,
            on=groupby_cols,
        )
        distance_df = _great_circle_distance(coordinates_df)
        radius_df = distance_df.groupby(groupby_cols).agg(
            sqrt(pys_sum(col("radius") ** 2 / col("num_records").cast("float"))).alias(
                "radius_of_gyration"
            )
        )
        if pivot_cols:
            aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
                "radius_of_gyration",
                cols_to_use_for_pivot=pivot_cols,
                agg_func=pys_sum,
            )
            pivoted_df = radius_df.groupby("caller_id").agg(*aggs)
        else:
            pivoted_df = radius_df.select("caller_id", "radius_of_gyration")

        if drop_cols:
            pivoted_df = pivoted_df.drop(*drop_cols)

        return pivoted_df

    base_cols = ["caller_id"]
    base_pivots: list[AllowedPivotColumnsEnum] = []
    dimensions = {
        "is_weekend": [True, False],
        "is_daytime": [True, False],
    }
    drop_cols = {
        "is_weekend": ["is_daytime"],
        "is_daytime": ["is_weekend"],
    }
    pivot_cols = {
        "is_weekend": [AllowedPivotColumnsEnum.IS_WEEKEND],
        "is_daytime": [AllowedPivotColumnsEnum.IS_DAYTIME],
    }
    meshgrid_for_dimensions = (
        np.array(np.meshgrid(*dimensions.values())).reshape(len(dimensions), -1).T
    )
    dfs_to_join = []
    for setting in meshgrid_for_dimensions:
        groupby_cols = base_cols.copy()
        cols_to_use_for_pivot = base_pivots.copy()
        cols_to_drop = []

        for i, (dim_name, _) in enumerate(dimensions.items()):
            if setting[i]:
                groupby_cols.append(dim_name)
                cols_to_use_for_pivot.extend(pivot_cols[dim_name])
            else:
                cols_to_drop.extend(drop_cols[dim_name])
        df = _get_groupby_and_pivot_df(
            groupby_cols, cols_to_use_for_pivot, cols_to_drop
        )

        dfs_to_join.append(df)
    return reduce(
        lambda left, right: left.join(right, on="caller_id", how="inner"), dfs_to_join
    )


def get_pareto_principle_antennas(
    spark_df: SparkDataFrame, percentage_threshold: float = 0.8
) -> SparkDataFrame:
    """
    The Pareto principle (80/20 rule) states that roughly 80% of effects come from 20% of causes.
    This function calculates the fraction of antennas that account for `threshold` percentage of
    a caller's interactions, disaggregated by weekday/weekend and daytime/nighttime interactions.

    Args:
        spark_df: Dataframe with 'caller_id', 'caller_antenna_id', 'is_daytime', 'is_weekend' columns
        percentage_threshold: The percentage threshold to calculate the Pareto principle for (default is 0.8 for 80%)

    Returns:
        df: Dataframe with Pareto principle antennas column
    """
    logger.info("Calculating Pareto principle antennas per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    def _get_groupby_and_pivot_df(
        groupby_cols: list[str],
        pivot_cols: list[AllowedPivotColumnsEnum],
        drop_cols: list[str] = [],
    ):
        # Configure windows for calculations
        window_1 = Window.partitionBy(*groupby_cols)
        window_2 = Window.partitionBy(*groupby_cols).orderBy(
            col("interaction_count").desc()
        )
        window_3 = Window.partitionBy(*groupby_cols).orderBy("row_number")

        # Calculate Pareto principle antenna stats
        antenna_df = (
            spark_df.groupby("caller_antenna_id", *groupby_cols)
            .agg(count(lit(0)).alias("interaction_count"))
            .withColumn("total_count", pys_sum("interaction_count").over(window_1))
            .withColumn("row_number", row_number().over(window_2))
            .withColumn("cumsum_count", pys_sum("interaction_count").over(window_3))
            .withColumn(
                "fraction_count", col("cumsum_count") / col("total_count").cast("float")
            )
            .withColumn(
                "row_number",
                when(col("fraction_count") >= percentage_threshold, col("row_number")),
            )
            .groupby(*groupby_cols)
            .agg(pys_min("row_number").alias("num_pareto_principle_antennas"))
        )

        if pivot_cols:
            aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
                "num_pareto_principle_antennas",
                cols_to_use_for_pivot=pivot_cols,
                agg_func=pys_sum,
            )
            pivoted_df = antenna_df.groupby("caller_id").agg(*aggs)
        else:
            pivoted_df = antenna_df.select("caller_id", "num_pareto_principle_antennas")
        return pivoted_df

    base_cols = ["caller_id"]
    base_pivots: list[AllowedPivotColumnsEnum] = []
    dimensions = {
        "is_weekend": [True, False],
        "is_daytime": [True, False],
    }
    drop_cols = {
        "is_weekend": ["is_daytime"],
        "is_daytime": ["is_weekend"],
    }
    pivot_cols = {
        "is_weekend": [AllowedPivotColumnsEnum.IS_WEEKEND],
        "is_daytime": [AllowedPivotColumnsEnum.IS_DAYTIME],
    }
    meshgrid_for_dimensions = (
        np.array(np.meshgrid(*dimensions.values())).reshape(len(dimensions), -1).T
    )
    dfs_to_join = []
    for setting in meshgrid_for_dimensions:
        groupby_cols = base_cols.copy()
        cols_to_use_for_pivot = base_pivots.copy()
        cols_to_drop = []
        if setting.sum() == 0:
            df = _get_groupby_and_pivot_df(groupby_cols, cols_to_use_for_pivot)
        else:
            for i, (dim_name, _) in enumerate(dimensions.items()):
                if setting[i]:
                    groupby_cols.append(dim_name)
                    cols_to_use_for_pivot.extend(pivot_cols[dim_name])
                else:
                    cols_to_drop.extend(drop_cols[dim_name])
            df = _get_groupby_and_pivot_df(
                groupby_cols, cols_to_use_for_pivot, cols_to_drop
            )
        dfs_to_join.append(df)
    return reduce(
        lambda left, right: left.join(right, on="caller_id", how="inner"), dfs_to_join
    )


def get_average_num_of_interactions_from_home_antennas(
    spark_df: SparkDataFrame,
) -> SparkDataFrame:
    """
    Get percentage of interactions from home antennas per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'caller_antenna_id', 'is_daytime', 'is_weekend' columns

    Returns:
        df: Dataframe with percentage of interactions from home antennas column
    """
    logger.info(
        "Calculating average number of interactions from home antennas per caller"
    )

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    # Identify home antenna per caller:
    # home antenna is the antenna from which the most nightime-calls are made
    window = Window.partitionBy("caller_id").orderBy(
        col("filtered_interaction_count").desc()
    )
    home_antenna_df = (
        spark_df.where(col("is_daytime") == 0)
        .groupby("caller_id", "caller_antenna_id")
        .agg(count(lit(0)).alias("filtered_interaction_count"))
        .withColumn("row_number", row_number().over(window))
        .where(col("row_number") == 1)
        .withColumnRenamed("caller_antenna_id", "home_antenna_id")
        .drop("filtered_interaction_count")
    )

    home_interaction_df = (
        spark_df.join(home_antenna_df, on="caller_id", how="inner")
        .withColumn(
            "is_home_interaction",
            when(col("caller_antenna_id") == col("home_antenna_id"), 1).otherwise(0),
        )
        .groupby("caller_id", "is_weekend", "is_daytime")
        .agg(pys_mean("is_home_interaction").alias("mean_home_antenna_interaction"))
    )

    aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
        "mean_home_antenna_interaction",
        cols_to_use_for_pivot=[
            AllowedPivotColumnsEnum.IS_WEEKEND,
            AllowedPivotColumnsEnum.IS_DAYTIME,
        ],
        agg_func=first,
    )

    return home_interaction_df.groupby("caller_id").agg(*aggs)


# International features
def get_international_interaction_statistics(
    spark_df: SparkDataFrame,
) -> SparkDataFrame:
    """
    Get number of international interactions per caller in the dataframe, disaggregated by transaction type.

    Args:
        spark_df: Dataframe with 'caller_id', 'transaction_type', 'transaction_scope', 'day' and 'duration' columns

    Returns:
        df: Dataframe with international transaction statistics per transaction type: number of recipients, number of unique recipients, number of unique days, total call duration, etc.
    """
    logger.info("Calculating international interaction statistics per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)

    international_df = spark_df.filter(
        col("transaction_scope") == TransactionScope.INTERNATIONAL.value
    )
    all_stats_df = international_df.groupBy("caller_id", "transaction_type").agg(
        count("recipient_id").alias("num_interactions"),
        countDistinct("recipient_id").alias("num_unique_recipients"),
        pys_sum("duration").alias("total_call_duration"),
        countDistinct("day").alias("num_unique_days"),
    )
    all_aggs = []
    for pivot_col in [
        "num_interactions",
        "num_unique_recipients",
        "total_call_duration",
        "num_unique_days",
    ]:
        aggs = _get_agg_columns_by_cdr_time_and_transaction_type(
            pivot_col,
            cols_to_use_for_pivot=[
                AllowedPivotColumnsEnum.TRANSACTION_TYPE,
            ],
            agg_func=pys_sum,
        )
        all_aggs.extend(aggs)
    stats_df = all_stats_df.groupby("caller_id").agg(*all_aggs)

    # Drop call duration columns for texts
    stats_df = stats_df.drop("text_total_call_duration")
    return stats_df


# TODO: this is a reimplementaion of deprecated.featurizer.location_features. However, there are additional calculations
# e.g. percentage unique callers per region, that seem incompatible with the per-caller
# structure we have followed so far. So current implementation does not mirror the deprecated one exactly.
def get_caller_counts_per_region(
    spark_df: SparkDataFrame, spark_antenna_df: SparkDataFrame
) -> SparkDataFrame:
    """
    Get location features per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id' column
        spark_antenna_df: Dataframe with 'caller_antenna_id', 'latitude', 'longitude' and 'region' columns

    Returns:
        df: Dataframe with location features columns
    """
    logger.info("Calculating caller counts per region")

    # Validate input dataframe
    validate_dataframe(spark_df, CallDataRecordTagged)
    validate_dataframe(spark_antenna_df, AntennaDataGeometryWithRegion)

    # Merge CDR and antenna data by caller ID
    joined_df = spark_df.join(spark_antenna_df, on="caller_antenna_id", how="inner")

    # Get caller statistics by region
    region_counts_df = joined_df.groupby("caller_id", "region").agg(
        countDistinct("recipient_id").alias("num_unique_interactions"),
        countDistinct("caller_antenna_id").alias("num_unique_antennas"),
    )

    # Pivot region statistics
    pivoted_df = (
        region_counts_df.groupby("caller_id")
        .pivot("region")
        .agg(first("num_unique_interactions"), first("num_unique_antennas"))
    )

    return pivoted_df


# Mobile data features
def get_mobile_data_stats(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Get mobile data usage statistics per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'day', 'volume' columns

    Returns:
        df: Dataframe with mobile data usage statistics columns
    """
    logger.info("Calculating mobile data usage statistics per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, MobileDataUsageDataWithDay)
    summary_stats_aggs = _get_summary_stats_cols(
        "volume",
        [
            StatsComputationMethodEnum.MEAN,
            StatsComputationMethodEnum.MIN,
            StatsComputationMethodEnum.MAX,
            StatsComputationMethodEnum.STD,
        ],
    )

    summary_stats_df = spark_df.groupby("caller_id").agg(
        pys_sum("volume").alias("total_data_volume"),
        countDistinct("day").alias("num_unique_days_with_data_usage"),
        *summary_stats_aggs,
    )
    return summary_stats_df


# Mobile money features
def get_mobile_money_amount_stats(
    spark_df: SparkDataFrame,
) -> SparkDataFrame:
    """
    Get mobile money amount statistics per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'primary_id', 'correspondent_id', 'transaction_type', 'amount', 'day' columns

    Returns:
        df: Dataframe with mobile money transaction statistics columns
    """
    logger.info("Calculating mobile money amount statistics per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, MobileMoneyDataWithDirection)

    summary_stats_cols = [
        StatsComputationMethodEnum.MEAN,
        StatsComputationMethodEnum.MIN,
        StatsComputationMethodEnum.MAX,
        StatsComputationMethodEnum.STD,
    ]
    summary_stats_aggs = _get_summary_stats_cols("amount", summary_stats_cols)

    summary_stats_all = spark_df.groupby("primary_id").agg(
        *summary_stats_aggs,
    )

    summary_stats_df = spark_df.groupby("primary_id", "transaction_type").agg(
        *summary_stats_aggs,
    )
    summary_stats_cols = [col for col in summary_stats_df.columns if "amount" in col]
    pivot_df = (
        summary_stats_df.groupby("primary_id")
        .pivot("transaction_type")
        .agg(*[first(col_name) for col_name in summary_stats_cols])
    )
    pivot_df = pivot_df.join(summary_stats_all, on="primary_id", how="inner")

    return pivot_df


def get_mobile_money_transaction_stats(
    spark_df: SparkDataFrame,
) -> SparkDataFrame:
    """
    Get mobile money transaction statistics per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'primary_id', 'correspondent_id', 'transaction_type' columns

    Returns:
        df: Dataframe with mobile money transaction statistics columns
    """
    logger.info("Calculating mobile money transaction statistics per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, MobileMoneyDataWithDirection)

    summary_stats_df = spark_df.groupby("primary_id", "transaction_type").agg(
        count("correspondent_id").alias("num_transactions"),
        countDistinct("correspondent_id").alias("num_unique_correspondents"),
    )
    pivot_df = (
        summary_stats_df.groupby("primary_id")
        .pivot("transaction_type")
        .agg(first("num_transactions"), first("num_unique_correspondents"))
    )

    return pivot_df


def get_mobile_money_balance_stats(
    spark_df: SparkDataFrame,
) -> SparkDataFrame:
    """
    Get mobile money balance statistics per caller in the dataframe.

    Args:
        spark_df: Dataframe with 'primary_id', 'transaction_type', 'balance_before', and 'balance_after' columns

    Returns:
        df: Dataframe with mobile money balance statistics columns
    """
    logger.info("Calculating mobile money balance statistics per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, MobileMoneyDataWithDirection)

    summary_stats_cols = [
        StatsComputationMethodEnum.MEAN,
        StatsComputationMethodEnum.MIN,
        StatsComputationMethodEnum.MAX,
        StatsComputationMethodEnum.STD,
    ]
    summary_stats_aggs = _get_summary_stats_cols(
        "balance_after", summary_stats_cols
    ) + _get_summary_stats_cols("balance_before", summary_stats_cols)

    summary_stats_all = spark_df.groupby("primary_id").agg(
        *summary_stats_aggs,
    )

    summary_stats_by_type = spark_df.groupby("primary_id", "transaction_type").agg(
        *summary_stats_aggs,
    )
    summary_stats_cols = [
        col for col in summary_stats_by_type.columns if "balance" in col
    ]
    pivot_df = (
        summary_stats_by_type.groupby("primary_id")
        .pivot("transaction_type")
        .agg(*[first(col_name) for col_name in summary_stats_cols])
    )
    pivot_df = pivot_df.join(summary_stats_all, on="primary_id", how="inner")

    return pivot_df


# Recharges features
def get_recharge_amount_stats(spark_df: SparkDataFrame) -> SparkDataFrame:
    """
    Get recharge amount statistics per user in the dataframe.

    Args:
        spark_df: Dataframe with 'caller_id', 'amount' columns

    Returns:
        df: Dataframe with recharge amount statistics columns
    """
    logger.info("Calculating recharge amount statistics per caller")

    # Validate input dataframe
    validate_dataframe(spark_df, RechargeDataWithDay)

    summary_stats_cols = _get_summary_stats_cols(
        "amount",
        [
            StatsComputationMethodEnum.MEAN,
            StatsComputationMethodEnum.MIN,
            StatsComputationMethodEnum.MAX,
            StatsComputationMethodEnum.STD,
        ],
    )

    summary_stats_df = spark_df.groupby("caller_id").agg(
        pys_sum("amount").alias("total_recharge_amount"),
        count("amount").alias("num_recharges"),
        countDistinct("day").alias("num_unique_recharge_days"),
        *summary_stats_cols,
    )
    return summary_stats_df


def preprocess_data(
    data_dict: dict[type[BaseModel], PandasDataFrame],
    filter_start_date: datetime,
    filter_end_date: datetime,
    spammer_threshold: float = 1.75,
    outlier_day_z_score_threshold: float = 2.0,
) -> SparkDataFrame:
    """
    Preprocess all data before featurizing

    Args:
        data_dict: Dictionary of dataframes with their corresponding schema types
        filter_start_date: Start date for filtering data
        filter_end_date: End date for filtering data
        spammer_threshold: Threshold for number of calls for identifying spammers
        outlier_day_z_score_threshold: z-score threshold for transactions, to identify days with unusual activity
    Returns:
        df: Dataframe with full set of features columns
    """
    logger.info("Preprocessing data")

    preprocessed_data: dict[type[BaseModel], PandasDataFrame] = {}
    spammers_list: list[str] = []

    for schema in [
        CallDataRecordData,
        MobileDataUsageData,
        MobileMoneyTransactionData,
        RechargeData,
    ]:
        validate_dataframe(data_dict[schema], schema, check_data_points=False)

        logger.info(f"Preprocessing data for schema: {schema.__name__}")
        # Filter to datetime
        filtered_df = filter_to_datetime(
            data_dict[schema],
            pd.to_datetime(filter_start_date),
            pd.to_datetime(filter_end_date),
        )

        if schema == CallDataRecordData:
            spammers_list = get_spammers_from_cdr_data(filtered_df, spammer_threshold)

        # Remove spammers
        filtered_no_spammers_df = filtered_df[
            ~filtered_df["caller_id"].isin(spammers_list)
        ]
        if "recipient_id" in filtered_df.columns:
            filtered_no_spammers_df = filtered_no_spammers_df[
                ~filtered_no_spammers_df["recipient_id"].isin(spammers_list)
            ]

        # Remove outlier days
        if schema == CallDataRecordData:
            outlier_days = get_outlier_days_from_cdr_data(
                filtered_no_spammers_df, outlier_day_z_score_threshold
            )
        filtered_no_outlier_days_df = filtered_no_spammers_df[
            ~filtered_no_spammers_df.timestamp.dt.date.isin(outlier_days)
        ]

        preprocessed_data[schema] = filtered_no_outlier_days_df

    return preprocessed_data


def featurize_cdr_data(
    cdr_data: PandasDataFrame,
    antenna_data: PandasDataFrame,
    max_wait_for_convo_in_seconds: int = 3600,
    pareto_threshold: float = 0.8,
) -> PandasDataFrame:
    """
    Retrieve all features for CDR data

    Args:
        cdr_data: Call record data
        antenna_data: Antenna data
        max_wait_for_convo_in_seconds: Maximum wait time between calls/texts to be considered part of the same conversation
        pareto_threshold: Threshold for Pareto principle calculations

    Returns:
        pandas dataframe containing the full set of features

    """
    # Validate dataframes
    validate_dataframe(cdr_data, CallDataRecordData)
    validate_dataframe(antenna_data, AntennaData)
    assert "region" in antenna_data.columns, "Antenna data must contain 'region' column"

    spark_session = get_spark_session()

    # Prepare CDR data: identify daytime/weekend, tag conversations
    spark_cdr = spark_session.createDataFrame(cdr_data)
    spark_antennas = spark_session.createDataFrame(antenna_data).withColumnRenamed(
        "antenna_id", "caller_antenna_id"
    )

    logger.info("Identifying daytime and weekend interactions")
    spark_cdr_with_daytime = identify_daytime(spark_cdr)
    spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)

    # Swap caller and recipient to get recipient-centric view
    logger.info("Swapping caller and recipient IDs for recipient-centric view")
    spark_cdr_swapped_caller_recipient = swap_caller_and_recipient(
        spark_cdr_with_weekend
    )

    # Identify and tag conversations
    logger.info("Identifying and tagging conversations")
    spark_cdr_tagged_conversations = identify_and_tag_conversations(
        spark_cdr_swapped_caller_recipient, max_wait=max_wait_for_convo_in_seconds
    )

    # Featurize CDR data
    spark_cdr_active_days = get_active_days(spark_cdr_tagged_conversations)
    spark_cdr_number_of_contacts_per_caller = get_number_of_contacts_per_caller(
        spark_cdr_tagged_conversations
    )
    spark_cdr_call_duration_stats = get_call_duration_stats(
        spark_cdr_tagged_conversations
    )
    spark_cdr_nocturnal_interactions = get_percentage_of_nocturnal_interactions(
        spark_cdr_tagged_conversations
    )
    spark_cdr_percentage_of_initiated_interactions = (
        get_percentage_of_initiated_conversations(spark_cdr_tagged_conversations)
    )
    spark_percentage_initiated_calls = get_percentage_of_initiated_calls(
        spark_cdr_tagged_conversations
    )
    spark_cdr_response_time_delay_stats = get_text_response_time_delay_stats(
        spark_cdr_tagged_conversations
    )
    spark_cdr_text_response_rate = get_text_response_rate(
        spark_cdr_tagged_conversations
    )
    spark_cdr_entropy_of_interactions = get_entropy_of_interactions_per_caller(
        spark_cdr_tagged_conversations
    )
    spark_cdr_outgoing_interactions_fraction = get_outgoing_interaction_fraction_stats(
        spark_cdr_tagged_conversations
    )
    spark_cdr_interaction_stats_per_caller = get_interaction_stats_per_caller(
        spark_cdr_tagged_conversations
    )
    spark_cdr_inter_event_time_stats = get_inter_event_time_stats(
        spark_cdr_tagged_conversations
    )
    spark_cdr_pareto_interaction_stats = get_pareto_principle_interaction_stats(
        spark_cdr_tagged_conversations, pareto_threshold
    )
    spark_cdr_pareto_call_duration_stats = get_pareto_principle_call_duration_stats(
        spark_cdr_tagged_conversations, pareto_threshold
    )
    spark_cdr_number_of_transactions = get_number_of_interactions_per_user(
        spark_cdr_tagged_conversations
    )
    spark_cdr_number_of_antennas = get_number_of_antennas(
        spark_cdr_tagged_conversations
    )
    spark_cdr_entropy_of_antennas = get_entropy_of_antennas_per_caller(
        spark_cdr_tagged_conversations
    )
    spark_cdr_radius_of_gyration = get_radius_of_gyration(
        spark_cdr_tagged_conversations, spark_antennas
    )
    spark_cdr_pareto_antennas = get_pareto_principle_antennas(
        spark_cdr_tagged_conversations, pareto_threshold
    )
    spark_cdr_home_antenna_interactions = (
        get_average_num_of_interactions_from_home_antennas(
            spark_cdr_tagged_conversations
        )
    )
    spark_cdr_international_stats = get_international_interaction_statistics(
        spark_cdr_tagged_conversations
    )
    spark_cdr_antenna_location_features = get_caller_counts_per_region(
        spark_cdr_tagged_conversations, spark_antennas
    )

    # Merge all features into a single dataframe on caller_id
    feature_dfs = [
        spark_cdr_active_days,
        spark_cdr_number_of_contacts_per_caller,
        spark_cdr_call_duration_stats,
        spark_cdr_nocturnal_interactions,
        spark_cdr_percentage_of_initiated_interactions,
        spark_percentage_initiated_calls,
        spark_cdr_response_time_delay_stats,
        spark_cdr_text_response_rate,
        spark_cdr_entropy_of_interactions,
        spark_cdr_outgoing_interactions_fraction,
        spark_cdr_interaction_stats_per_caller,
        spark_cdr_inter_event_time_stats,
        spark_cdr_pareto_interaction_stats,
        spark_cdr_pareto_call_duration_stats,
        spark_cdr_number_of_transactions,
        spark_cdr_number_of_antennas,
        spark_cdr_entropy_of_antennas,
        spark_cdr_radius_of_gyration,
        spark_cdr_pareto_antennas,
        spark_cdr_home_antenna_interactions,
        spark_cdr_international_stats,
        spark_cdr_antenna_location_features,
    ]
    spark_merged_df = reduce(
        lambda df1, df2: df1.join(df2, on="caller_id", how="outer"),
        feature_dfs,
    )

    return spark_merged_df.toPandas()


def featurize_mobile_data_usage_data(
    mobile_data: PandasDataFrame,
) -> PandasDataFrame:
    """
    Retrieve all features for mobile data usage

    Args:
        mobile_data: Mobile data usage records

    Returns:
        pandas dataframe containing the full set of features
    """
    # Validate dataframe
    validate_dataframe(mobile_data, MobileDataUsageData)

    spark_session = get_spark_session()
    spark_mobile_data = spark_session.createDataFrame(mobile_data)

    spark_mobile_data_stats = get_mobile_data_stats(spark_mobile_data)

    return spark_mobile_data_stats.toPandas()


def featurize_mobile_money_data(
    mobile_money_data: PandasDataFrame,
) -> PandasDataFrame:
    """
    Retrieve all features for mobile money data

    Args:
        mobile_money_data: Mobile money transaction records

    Returns:
        pandas dataframe containing the full set of features
    """
    # Validate dataframe
    validate_dataframe(mobile_money_data, MobileMoneyTransactionData)

    spark_session = get_spark_session()
    spark_mobile_money_data = spark_session.createDataFrame(mobile_money_data)

    logger.info("Identifying direction of mobile money transactions")
    spark_mobile_money_with_direction = identify_mobile_money_transaction_direction(
        spark_mobile_money_data
    )
    spark_mobile_money_amount_stats = get_mobile_money_amount_stats(
        spark_mobile_money_with_direction
    )
    spark_mobile_money_transaction_stats = get_mobile_money_transaction_stats(
        spark_mobile_money_with_direction
    )
    spark_mobile_money_balance_stats = get_mobile_money_balance_stats(
        spark_mobile_money_with_direction
    )

    # Merge all features into a single dataframe on primary_id
    feature_dfs = [
        spark_mobile_money_amount_stats,
        spark_mobile_money_transaction_stats,
        spark_mobile_money_balance_stats,
    ]
    spark_merged_df = reduce(
        lambda df1, df2: df1.join(df2, on="primary_id", how="outer"),
        feature_dfs,
    )
    spark_merged_df = spark_merged_df.withColumnRenamed("primary_id", "caller_id")

    return spark_merged_df.toPandas()


def featurize_recharge_data(
    recharge_data: PandasDataFrame,
) -> PandasDataFrame:
    """
    Retrieve all features for recharge data

    Args:
        recharge_data: Recharge records
    Returns:
        pandas dataframe containing the full set of features
    """
    # Validate dataframe
    validate_dataframe(recharge_data, RechargeData)

    spark_session = get_spark_session()
    spark_recharge_data = spark_session.createDataFrame(recharge_data)

    spark_recharge_amount_stats = get_recharge_amount_stats(spark_recharge_data)

    return spark_recharge_amount_stats.toPandas()


def featurize_all_data(
    preprocessed_data: dict[type[BaseModel], PandasDataFrame],
    max_wait_for_convo_in_seconds: int = 3600,
    pareto_threshold: float = 0.8,
) -> PandasDataFrame:
    """
    Featurize all preprocessed data

    Args:
        preprocessed_data: Dictionary of preprocessed dataframes with their corresponding schema types
        max_wait_for_convo_in_seconds: Maximum wait time between calls/texts to be considered part of the same conversation
        pareto_threshold: Threshold for Pareto principle calculations

    Returns:
        pandas dataframe containing the full set of features
    """
    logger.info("Featurizing CDR data")
    cdr_features_df = featurize_cdr_data(
        preprocessed_data[CallDataRecordData],
        preprocessed_data[AntennaData],
        max_wait_for_convo_in_seconds,
        pareto_threshold,
    )

    logger.info("Featurizing mobile data usage data")
    mobile_data_features_df = featurize_mobile_data_usage_data(
        preprocessed_data[MobileDataUsageData]
    )

    logger.info("Featurizing mobile money data")
    mobile_money_features_df = featurize_mobile_money_data(
        preprocessed_data[MobileMoneyTransactionData]
    )

    logger.info("Featurizing recharge data")
    recharge_features_df = featurize_recharge_data(preprocessed_data[RechargeData])

    # Merge all features into a single dataframe on caller_id
    logger.info("Merging all features into a single dataframe")
    feature_dfs = [
        cdr_features_df,
        mobile_data_features_df,
        mobile_money_features_df,
        recharge_features_df,
    ]
    merged_df = reduce(
        lambda df1, df2: pd.merge(df1, df2, on="caller_id", how="inner"),
        feature_dfs,
    )

    return merged_df
