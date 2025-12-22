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

from .schemas import GetHomeLocationAlgorithm, GeographicUnit
import pandas as pd
import geopandas as gpd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    count,
    desc_nulls_last,
    row_number,
    col,
    to_date,
    countDistinct,
)
from pyspark.sql.window import Window


# Prepare data for home location inference
def _prepare_home_location_data(
    validated_cdr_data: pd.DataFrame,
    validated_antenna_data: pd.DataFrame,
    geographic_unit: GeographicUnit,
    shapefile_data: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    """
    Prepare data for home location inference

    Args:
        validated_cdr_data: validated call data records
        validated_antenna_data: validated antenna data
        geographic_unit: geographic unit to use for home location inference

    Returns:
        prepared_data: prepared data for home location inference
    """
    columns_to_drop = []
    match geographic_unit:
        case GeographicUnit.ANTENNA_ID:
            prepared_data = validated_cdr_data.merge(
                validated_antenna_data,
                left_on="caller_antenna_id",
                right_on="antenna_id",
                how="inner",
            )
            columns_to_drop = ["antenna_id"]

        case GeographicUnit.TOWER_ID:
            if "tower_id" not in validated_antenna_data.columns:
                raise ValueError(
                    "Antenna data must contain 'tower_id' column for geographic unit TOWER_ID."
                )
            prepared_data = validated_cdr_data.merge(
                validated_antenna_data,
                left_on="caller_antenna_id",
                right_on="tower_id",
                how="inner",
            )
            columns_to_drop = ["tower_id"]

        case GeographicUnit.SHAPEFILE:
            if shapefile_data is None:
                raise ValueError(
                    f"Shapefile data must be provided for geographic unit {GeographicUnit.SHAPEFILE}."
                )

            antennas_gdf = gpd.GeoDataFrame(
                validated_antenna_data,
                geometry=gpd.points_from_xy(
                    validated_antenna_data.longitude, validated_antenna_data.latitude
                ),
                crs="EPSG:4326",
            )
            antennas = gpd.sjoin(
                antennas_gdf, shapefile_data, predicate="within", how="left"
            )

            prepared_data = validated_cdr_data.merge(
                antennas,
                left_on="caller_antenna_id",
                right_on="antenna_id",
                how="inner",
            )
            columns_to_drop = ["antenna_id", "geometry"]

        case _:
            raise ValueError(f"Unsupported geographic unit: {geographic_unit}")

    if prepared_data.empty:
        raise ValueError(
            "Prepared data is empty after merging CDR and antenna data. Please check the input data."
        )
    prepared_data.drop(columns=columns_to_drop, inplace=True)

    return prepared_data


def _infer_home_locations(
    prepared_data: pd.DataFrame,
    algorithm: GetHomeLocationAlgorithm,
    spark_session: SparkSession,
    additional_columns_to_keep: list[str] = [],
) -> pd.DataFrame:
    """
    Infer home locations based on the specified algorithm

    Args:
        prepared_data: prepared data for home location inference
        algorithm: algorithm to use for home location inference
        spark_session: Spark session
        additional_columns_to_keep: list of additional columns to keep in the output
            (by default we keep only caller_id and caller_antenna_id columns)

    Returns:
        home_locations: inferred home locations
    """
    prepared_data_spark = spark_session.createDataFrame(prepared_data)
    if not set(additional_columns_to_keep).issubset(set(prepared_data_spark.columns)):
        raise ValueError(
            "Some additional columns to keep are not present in the prepared data."
        )

    match algorithm:
        case GetHomeLocationAlgorithm.COUNT_TRANSACTIONS:
            grouped_data = prepared_data_spark.groupby(
                [
                    "caller_id",
                    "caller_antenna_id",
                ]
                + additional_columns_to_keep
            ).agg(
                count("timestamp").alias(
                    GetHomeLocationAlgorithm.COUNT_TRANSACTIONS.value
                )
            )

            window = Window.partitionBy("caller_id").orderBy(
                desc_nulls_last(GetHomeLocationAlgorithm.COUNT_TRANSACTIONS.value)
            )
            grouped_data = (
                grouped_data.withColumn("order", row_number().over(window))
                .where(col("order") == 1)
                .select(
                    [
                        "caller_id",
                        "caller_antenna_id",
                        GetHomeLocationAlgorithm.COUNT_TRANSACTIONS.value,
                    ]
                    + additional_columns_to_keep
                )
            )

        case GetHomeLocationAlgorithm.COUNT_DAYS:
            prepared_data_spark = prepared_data_spark.withColumn(
                "day", to_date("timestamp")
            )
            grouped_data = prepared_data_spark.groupby(
                [
                    "caller_id",
                    "caller_antenna_id",
                ]
                + additional_columns_to_keep
            ).agg(countDistinct("day").alias(GetHomeLocationAlgorithm.COUNT_DAYS.value))
            window = Window.partitionBy("caller_id").orderBy(
                desc_nulls_last(GetHomeLocationAlgorithm.COUNT_DAYS.value)
            )
            grouped_data = (
                grouped_data.withColumn("order", row_number().over(window))
                .where(col("order") == 1)
                .select(
                    [
                        "caller_id",
                        "caller_antenna_id",
                        GetHomeLocationAlgorithm.COUNT_DAYS.value,
                    ]
                    + additional_columns_to_keep
                )
            )

        case GetHomeLocationAlgorithm.COUNT_MODAL_DAYS:
            prepared_data_spark = prepared_data_spark.withColumn(
                "day", to_date("timestamp")
            )
            grouped_data = prepared_data_spark.groupby(
                ["caller_id", "caller_antenna_id", "day"] + additional_columns_to_keep
            ).agg(count("timestamp").alias("transactions_per_day_count"))
            window = Window.partitionBy(["caller_id", "day"]).orderBy(
                desc_nulls_last("transactions_per_day_count")
            )
            grouped_data = (
                grouped_data.withColumn("order", row_number().over(window))
                .where(col("order") == 1)
                .groupby(
                    ["caller_id", "caller_antenna_id"] + additional_columns_to_keep
                )
                .agg(
                    count("order").alias(
                        GetHomeLocationAlgorithm.COUNT_MODAL_DAYS.value
                    )
                )
            )
            window = Window.partitionBy("caller_id").orderBy(
                desc_nulls_last(GetHomeLocationAlgorithm.COUNT_MODAL_DAYS.value)
            )
            grouped_data = (
                grouped_data.withColumn("order", row_number().over(window))
                .where(col("order") == 1)
                .select(
                    [
                        "caller_id",
                        "caller_antenna_id",
                        GetHomeLocationAlgorithm.COUNT_MODAL_DAYS.value,
                    ]
                    + additional_columns_to_keep
                )
            )
        case _:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    grouped_data_pd = grouped_data.toPandas()
    return grouped_data_pd


def get_home_locations(
    spark_session: SparkSession,
    validated_cdr_data: pd.DataFrame,
    validated_antenna_data: pd.DataFrame,
    geographic_unit: GeographicUnit,
    algorithm: GetHomeLocationAlgorithm,
    shapefile_data: gpd.GeoDataFrame | None = None,
    additional_columns_to_keep: list[str] = [],
) -> pd.DataFrame:
    """
    Get home locations based on the specified parameters

    Args:
        spark_session: Spark session
        validated_cdr_data: validated CDR data
        validated_antenna_data: validated antenna data
        geographic_unit: geographic unit for home location inference
        algorithm: algorithm to use for home location inference
        shapefile_data: optional shapefile data for geographic boundaries
        additional_columns_to_keep: list of additional columns to keep in the output
            (by default we keep only caller_id and caller_antenna_id columns)

    Returns:
        DataFrame containing the inferred home locations
    """
    prepared_data = _prepare_home_location_data(
        validated_cdr_data, validated_antenna_data, geographic_unit, shapefile_data
    )
    home_locations = _infer_home_locations(
        prepared_data, algorithm, spark_session, additional_columns_to_keep
    )
    return home_locations


def get_accuracy(
    inferred_home_locations: pd.DataFrame,
    groundtruth_home_locations: pd.DataFrame,
    column_to_merge_on: str = "caller_id",
    column_to_measure_on: str = "caller_antenna_id",
) -> pd.DataFrame:
    """
    Get accuracy of inferred home locations compared to true home locations

    Args:
        geographic_unit: geographic unit for home location inference
        algorithm: algorithm used for home location inference
        column_to_merge_on: column to merge on (default is 'caller_id')
        column_to_measure_on: column to measure accuracy on (default is 'caller_antenna_id')

    Returns:
        table: DataFrame containing accuracy, precision and recall metrics
    """
    # Check for errors in input data
    if (column_to_merge_on not in inferred_home_locations.columns) or (
        column_to_merge_on not in groundtruth_home_locations.columns
    ):
        raise ValueError(
            f"Column '{column_to_merge_on}' must be present in both inferred and groundtruth home locations data."
        )
    if (column_to_measure_on not in inferred_home_locations.columns) or (
        column_to_measure_on not in groundtruth_home_locations.columns
    ):
        raise ValueError(
            f"Column '{column_to_measure_on}' must be present in both inferred and groundtruth home locations data."
        )

    # Merge data inferred and groundtruth home locations
    merged_data = inferred_home_locations.merge(
        groundtruth_home_locations,
        on=column_to_merge_on,
        how="inner",
        suffixes=("_inferred", "_groundtruth"),
    )

    # Calculate metrics
    merged_data["is_correct"] = (
        merged_data[column_to_measure_on + "_inferred"]
        == merged_data[column_to_measure_on + "_groundtruth"]
    )

    recall = (
        merged_data[[column_to_measure_on + "_groundtruth", "is_correct"]]
        .groupby(column_to_measure_on + "_groundtruth", as_index=False)
        .mean()
    )
    recall.rename(columns={"is_correct": "recall"}, inplace=True)
    precision = (
        merged_data[[column_to_measure_on + "_inferred", "is_correct"]]
        .groupby(column_to_measure_on + "_inferred", as_index=False)
        .mean()
    )
    precision.rename(columns={"is_correct": "precision"}, inplace=True)

    # Merge metrics into a single table
    table = recall.merge(
        precision,
        left_on=column_to_measure_on + "_groundtruth",
        right_on=column_to_measure_on + "_inferred",
        how="outer",
    ).fillna(0)

    return table
