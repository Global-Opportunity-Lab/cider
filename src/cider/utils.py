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

import types
from typing import get_origin
import warnings
from typing_extensions import get_args
from pydantic import BaseModel, ValidationError
from pyspark.sql import DataFrame as SparkDataFrame
from pandas import DataFrame as PandasDataFrame
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
from cider.schemas import (
    CallDataRecordData,
    AntennaData,
    MobileMoneyTransactionData,
    RechargeData,
    MobileDataUsageData,
    MobileMoneyTransactionType,
)
from shapely.geometry import box
from enum import Enum
from pyspark.sql import SparkSession
import logging

SPARK_SESSION: SparkSession | None = None


def setup_logger(name: str, logger_level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with the specified name.

    Args:
        name: Name of the logger

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logger_level)

    return logger


def get_spark_session():
    """
    Create a spark session for converting pandas dataframes into spark dataframes.
    """
    global SPARK_SESSION
    if not SPARK_SESSION or SPARK_SESSION._jsc.sc().isStopped():
        SPARK_SESSION = (
            SparkSession.builder.master("local[1]")
            .config("spark.driver.memory", "8g")
            .config("spark.sql.shuffle.partitions", 8)
            .appName("pytest-spark")
            .getOrCreate()
        )

    return SPARK_SESSION


def validate_dataframe(
    df: SparkDataFrame | PandasDataFrame,
    required_schema: type[BaseModel],
    check_data_points: bool = False,
) -> None:
    """
    Validate that the dataframe has the required schema.

    Args:
        df: Spark or Pandas dataframe to validate
        required_schema: Pydantic BaseModel schema that the dataframe must conform to
        check_data_points: Whether to check each of the rows in the dataframe, in addition to the column names. Default is False.

    Raises:
        ValueError: If any of the required columns are missing from the dataframe
    """
    df_columns = set(df.columns)
    required_columns = set(
        [k for k, field in required_schema.model_fields.items() if field.is_required()]
    )
    missing_columns = required_columns - df_columns
    if missing_columns:
        raise ValueError(
            f"The following required columns are missing from the dataframe: {missing_columns}"
        )

    if check_data_points:
        if isinstance(df, SparkDataFrame):
            pandas_df = df.toPandas()
        else:
            pandas_df = df.copy()

        for index, row in pandas_df.iterrows():
            try:
                required_schema.model_validate(row.to_dict())
            except ValidationError as e:
                raise ValueError(
                    f"Row {index} does not conform to the required schema: {e}"
                ) from e


# Synthetic data generation functions


def _get_column_types(
    schema: type[BaseModel], keep_optional_columns: bool
) -> dict[str, type]:
    """
    Get a mapping of column names to their types from a Pydantic BaseModel schema.

    Args:
        schema: Pydantic BaseModel schema to extract column types from
        keep_optional_columns: Whether to keep or discard optional types (i.e., Union types with NoneType)
    Returns:
        Dictionary mapping column names to their types
    """
    column_types = {}
    for key, value in schema.model_fields.items():
        origin = get_origin(value.annotation)
        if origin == types.UnionType:
            args = get_args(value.annotation)
            if keep_optional_columns and type(None) in args:
                # Pick the first argument that is not NoneType
                value_type = next(arg for arg in args if arg is not type(None))
            elif not keep_optional_columns and type(None) in args:
                continue  # keep_optional_columns is False, skip this column
            else:
                # If Union types are not none, simply take the first type
                value_type = next(arg for arg in get_args(value))
        else:
            value_type = value.annotation

        column_types[key] = value_type

    return column_types


def generate_synthetic_data(
    schema: type[BaseModel],
    num_data_points: int,
    random_seed: int = 42,
    keep_optional_columns: bool = False,
) -> PandasDataFrame:
    """
    Generate synthetic data for testing purposes.

    Args:
        schema: Pydantic BaseModel schema to generate data for
        num_data_points: Number of synthetic data points to generate
        random_seed: Random seed for reproducibility
        keep_optional_columns: Whether to include optional fields in the generated data

    Returns:
        Pandas DataFrame with synthetic CDR data
    """
    # Raise warnings for specific schemas
    if schema == AntennaData:
        warnings.warn(
            "Generating synthetic data for AntennaData schema may result in unrealistic latitude and longitude values. "
            "Use the `generate_antenna_data` function for more realistic antenna data.",
            Warning,
        )

    if schema == CallDataRecordData:
        warnings.warn(
            "Generating synthetic data for CallDataRecordData schema may result in unrealistic duration values and incompatible antenna values."
            "Use the `correct_generated_synthetic_cdr_data` function to correct these values after generation.",
            Warning,
        )

    if schema == MobileMoneyTransactionData:
        warnings.warn(
            "Generating synthetic data for MobileMoneyTransactionData schema may result in unrealistic amount values."
            "Use the `correct_generated_synthetic_mobile_money_transaction_data` function to correct these values after generation.",
            Warning,
        )

    column_types = _get_column_types(schema, keep_optional_columns)

    np.random.seed(random_seed)
    data = {}

    for key, value in column_types.items():
        if value is str:
            data[key] = [f"{key}_{i}" for i in range(num_data_points)]
        elif value is int:
            data[key] = np.random.randint(0, 1000, size=num_data_points).tolist()
        elif value is float:
            data[key] = np.random.uniform(0, 1000, size=num_data_points).tolist()
        elif value is datetime:
            start_date = pd.to_datetime("2023-01-01")
            data[key] = [
                start_date + pd.Timedelta(days=int(np.random.uniform(0, 365)))
                for _ in range(num_data_points)
            ]
        elif isinstance(value, type) and issubclass(value, Enum):
            enum_values = [e.value for e in list(value)]
            data[key] = np.random.choice(enum_values, size=num_data_points).tolist()
        else:
            raise ValueError(f"Unsupported column type: {value} for column {key}")

    return pd.DataFrame(data)


def correct_generated_synthetic_cdr_data(
    cdr_df: PandasDataFrame, num_unique_antenna_ids: int, random_seed: int = 42
) -> PandasDataFrame:
    """
    Correct synthetic CDR data for testing purposes.

    Args:
        cdr_df: Pandas DataFrame with synthetic CDR data
        num_unique_antenna_ids: Number of unique antenna IDs to use in the corrected data
        random_seed: Random seed for reproducibility
    Returns:
        Pandas DataFrame with corrected synthetic CDR data
    """
    validate_dataframe(
        cdr_df, required_schema=CallDataRecordData, check_data_points=True
    )

    # Ensure that duration for text data is zero
    cdr_df.loc[cdr_df["transaction_type"] == "text", "duration"] = 0.0

    np.random.seed(random_seed)

    # Redo antenna ids to be consistent
    unique_antenna_ids = [f"antenna_id_{i}" for i in range(num_unique_antenna_ids)]
    cdr_df["caller_antenna_id"] = np.random.choice(unique_antenna_ids, size=len(cdr_df))
    if "recipient_antenna_id" in cdr_df.columns:
        cdr_df["recipient_antenna_id"] = np.random.choice(
            unique_antenna_ids, size=len(cdr_df)
        )
    return cdr_df


def generate_antenna_data(num_antennas: int, random_seed: int = 42) -> PandasDataFrame:
    """
    Generate synthetic antenna data for testing purposes.

    Args:
        num_antennas: Number of unique antennas to generate
        random_seed: Random seed for reproducibility
    Returns:
        Pandas DataFrame with synthetic antenna data
    """
    np.random.seed(random_seed)
    antenna_ids = [f"antenna_id_{i}" for i in range(num_antennas)]
    tower_ids = [f"tower_id_{i}" for i in range(num_antennas)]
    latitudes = np.random.uniform(-90, 90, size=num_antennas).tolist()
    longitudes = np.random.uniform(-180, 180, size=num_antennas).tolist()

    data = {
        "antenna_id": antenna_ids,
        "tower_id": tower_ids,
        "latitude": latitudes,
        "longitude": longitudes,
    }

    return pd.DataFrame(data)


def correct_generated_synthetic_mobile_money_transaction_data(
    mobile_money_df: PandasDataFrame,
) -> PandasDataFrame:
    """
    Correct synthetic Mobile Money Transaction data for testing purposes.

    Args:
        mobile_money_df: Pandas DataFrame with synthetic Mobile Money Transaction data

    Returns:
        Pandas DataFrame with corrected synthetic Mobile Money Transaction data
    """
    validate_dataframe(
        mobile_money_df,
        required_schema=MobileMoneyTransactionData,
        check_data_points=False,
    )

    # Ensure cashin transactions have negative amounts
    mobile_money_df.loc[
        mobile_money_df["transaction_type"] == MobileMoneyTransactionType.CASHIN.value,
        "amount",
    ] = -abs(
        mobile_money_df.loc[
            mobile_money_df["transaction_type"]
            == MobileMoneyTransactionType.CASHIN.value,
            "amount",
        ]
    )

    # Ensure caller_balance_after matches caller_balance_before - amount
    mobile_money_df["caller_balance_after"] = (
        mobile_money_df["caller_balance_before"] - mobile_money_df["amount"]
    )

    if "recipient_id" in mobile_money_df.columns:

        # Ensure that recipient_id, recipient_balance_before, and recipient_balance_after are
        # None for cashin and cashout transactions
        mask = mobile_money_df["transaction_type"].isin(
            [
                MobileMoneyTransactionType.CASHIN.value,
                MobileMoneyTransactionType.CASHOUT.value,
            ]
        )
        mobile_money_df.loc[mask, "recipient_id"] = None
        mobile_money_df.loc[mask, "recipient_balance_before"] = None
        mobile_money_df.loc[mask, "recipient_balance_after"] = None

        # For transactions with recipient_id, ensure recipient_balance_after matches recipient_balance_before + amount
        mobile_money_df.loc[~mask, "recipient_balance_after"] = (
            mobile_money_df.loc[~mask, "recipient_balance_before"]
            + mobile_money_df.loc[~mask, "amount"]
        )

    return mobile_money_df


def generate_synthetic_shapefile(
    antenna_df: pd.DataFrame, num_regions: int, random_seed: int = 42
) -> gpd.GeoDataFrame:
    """
    Generate shapefile for regions based on antenna data

    Args:
        antenna_df: Synthetic antenna data
        num_regions: Number of regions to create
        random_seed: Random seed for reproducibility
    Returns:
        gpd.GeoDataFrame containing region names and corresponding shape geometry
    """
    validate_dataframe(antenna_df, AntennaData)

    # Get convex hull of antenna locations
    antenna_gdf = gpd.GeoDataFrame(
        antenna_df.antenna_id,
        geometry=gpd.points_from_xy(
            x=antenna_df["longitude"], y=antenna_df["latitude"]
        ),
    ).set_crs(epsg=4326)

    convex_hull = antenna_gdf.union_all().convex_hull.buffer(1.0)

    # Sample random points inside the polygon for tesselation
    np.random.seed(random_seed)
    minx, miny, maxx, maxy = convex_hull.bounds
    x_grid = np.sort(
        np.random.uniform(low=minx, high=maxx, size=int(np.ceil(np.sqrt(num_regions))))
    )
    x_grid = [minx] + list(x_grid) + [maxx]

    y_grid = np.sort(
        np.random.uniform(low=miny, high=maxy, size=int(np.ceil(np.sqrt(num_regions))))
    )
    y_grid = [miny] + list(y_grid) + [maxy]

    regions: dict[str, list] = {"region": [], "geometry": []}
    num_regions_found = 0
    for x1, x2 in zip(x_grid[:-1], x_grid[1:]):
        for y1, y2 in zip(y_grid[:-1], y_grid[1:]):
            if num_regions_found == num_regions:
                break
            if num_regions_found == (num_regions - 1):
                cell = box(x1, y1, maxx, maxy)
            else:
                cell = box(x1, y1, x2, y2)
            if convex_hull.intersects(cell):
                regions["region"].append(f"region_{num_regions_found}")
                regions["geometry"].append(cell.intersection(convex_hull))
                num_regions_found += 1

    regions_gdf = gpd.GeoDataFrame(
        {"region": regions["region"], "geometry": regions["geometry"]},
        crs="EPSG:4326",
    )
    return regions_gdf


def generate_all_synthetic_data(
    num_data_points: int,
    num_unique_antenna_ids: int,
    random_seed: int = 42,
) -> dict[type[BaseModel], PandasDataFrame]:
    """
    Generate synthetic data for all schemas for testing purposes.
    Args:
        num_data_points: Number of synthetic data points to generate for each schema
        num_unique_antenna_ids: Number of unique antenna IDs to use in the CDR data
        random_seed: Random seed for reproducibility
    Returns:
        Dictionary mapping schema names to Pandas DataFrames with synthetic data
    """
    synthetic_data = {}
    logger = setup_logger(__name__)

    # Generate synthetic CDR data
    logger.info("Generating synthetic call data record data")
    synthetic_cdr_df = generate_synthetic_data(
        schema=CallDataRecordData,
        num_data_points=num_data_points,
        random_seed=random_seed,
        keep_optional_columns=True,
    )

    logger.info("Correcting synthetic call data record data")
    synthetic_data[CallDataRecordData] = correct_generated_synthetic_cdr_data(
        synthetic_cdr_df,
        num_unique_antenna_ids,
        random_seed=random_seed,
    )

    # Generate synthetic Mobile Money Transaction data
    logger.info("Generating synthetic mobile money transaction data")
    synthetic_mobile_money_df = generate_synthetic_data(
        schema=MobileMoneyTransactionData,
        num_data_points=num_data_points,
        random_seed=random_seed,
        keep_optional_columns=True,
    )

    logger.info("Correcting synthetic mobile money transaction data")
    synthetic_data[MobileMoneyTransactionData] = (
        correct_generated_synthetic_mobile_money_transaction_data(
            synthetic_mobile_money_df
        )
    )

    # Generate synthetic Antenna data
    logger.info("Generating synthetic antenna data")
    synthetic_data[AntennaData] = generate_antenna_data(
        num_antennas=num_unique_antenna_ids, random_seed=random_seed
    )

    # Generate synthetic Recharge data
    logger.info("Generating synthetic recharge data")
    synthetic_data[RechargeData] = generate_synthetic_data(
        schema=RechargeData,
        num_data_points=num_data_points,
        random_seed=random_seed,
        keep_optional_columns=True,
    )

    # Generate synthetic Mobile Data Usage data
    logger.info("Generating synthetic mobile data usage data")
    synthetic_data[MobileDataUsageData] = generate_synthetic_data(
        schema=MobileDataUsageData,
        num_data_points=num_data_points,
        random_seed=random_seed,
        keep_optional_columns=True,
    )

    return synthetic_data
