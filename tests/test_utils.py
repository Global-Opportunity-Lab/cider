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


from cider.utils import (
    validate_dataframe,
    generate_synthetic_data,
    generate_antenna_data,
    correct_generated_synthetic_mobile_money_transaction_data,
    correct_generated_synthetic_cdr_data,
    generate_synthetic_shapefile,
)
import pytest
from conftest import CDR_DATA, ANTENNA_DATA
import pandas as pd
from cider.schemas import (
    CallDataRecordData,
    AntennaData,
    RechargeData,
    MobileMoneyTransactionData,
    MobileDataUsageData,
)
import numpy as np


def test_validate_dataframe(spark):
    # Create pandas DataFrame from test data
    pandas_df = pd.DataFrame(CDR_DATA).copy()

    # Create Spark DataFrame from test data
    spark_df = spark.createDataFrame(pandas_df)

    # Test that validation passes for correct schema
    assert (
        validate_dataframe(
            spark_df, required_schema=CallDataRecordData, check_data_points=True
        )
        is None
    )
    assert (
        validate_dataframe(
            pandas_df, required_schema=CallDataRecordData, check_data_points=True
        )
        is None
    )

    # Test that validation fails for missing required column
    pandas_df.drop(columns=["caller_id"], inplace=True)
    spark_df = spark.createDataFrame(pandas_df)
    with pytest.raises(
        ValueError,
        match="The following required columns are missing from the dataframe: {'caller_id'}",
    ):
        validate_dataframe(pandas_df, required_schema=CallDataRecordData)
    with pytest.raises(
        ValueError,
        match="The following required columns are missing from the dataframe: {'caller_id'}",
    ):
        validate_dataframe(spark_df, required_schema=CallDataRecordData)

    # Test that validation fails for incorrect data points
    cdr_incorrect_data = CDR_DATA.copy()
    cdr_incorrect_data["transaction_type"][0] = "invalid_type"
    df_incorrect_data = pd.DataFrame(cdr_incorrect_data)
    spark_df_incorrect_data = spark.createDataFrame(df_incorrect_data)
    with pytest.raises(ValueError):
        validate_dataframe(
            df_incorrect_data,
            required_schema=CallDataRecordData,
            check_data_points=True,
        )
    with pytest.raises(ValueError):
        validate_dataframe(
            spark_df_incorrect_data,
            required_schema=CallDataRecordData,
            check_data_points=True,
        )


@pytest.mark.parametrize(
    "schema,",
    [
        CallDataRecordData,
        AntennaData,
        RechargeData,
        MobileDataUsageData,
    ],
)
def test_generate_synthetic_data(schema):

    num_data_points = 100
    synthetic_df = generate_synthetic_data(
        schema=schema, num_data_points=num_data_points
    )

    # Validate the synthetic DataFrame against the schema and num datapoints
    assert len(synthetic_df) == num_data_points
    assert (
        validate_dataframe(
            synthetic_df,
            required_schema=schema,
            check_data_points=True,
        )
        is None
    )


@pytest.mark.parametrize(
    "schema",
    [
        CallDataRecordData,
        MobileMoneyTransactionData,
        AntennaData,
    ],
)
def test_generate_synthetic_data_raises_a_warning(schema):
    with pytest.warns(Warning):
        generate_synthetic_data(schema=schema, num_data_points=100)


def test_generate_synthetic_data_with_optional_columns():
    num_data_points = 100
    synthetic_df = generate_synthetic_data(
        schema=CallDataRecordData,
        num_data_points=num_data_points,
        keep_optional_columns=True,
    )

    # Validate the synthetic DataFrame against the schema and num datapoints
    assert len(synthetic_df) == num_data_points
    assert set(synthetic_df.columns) == set(CallDataRecordData.model_fields.keys())


def test_generate_antenna_data():
    num_antennas = 50
    antenna_df = generate_antenna_data(num_antennas=num_antennas)

    # Validate the synthetic DataFrame against the AntennaData schema and num antennas
    assert len(antenna_df) == num_antennas
    assert (
        validate_dataframe(
            antenna_df,
            required_schema=AntennaData,
            check_data_points=True,
        )
        is None
    )


@pytest.mark.parametrize(
    "keep_optional_columns",
    [True, False],
)
def test_correct_generated_synthetic_mobile_money_transaction_data(
    keep_optional_columns,
):
    num_data_points = 100
    synthetic_mobile_money_df = generate_synthetic_data(
        schema=MobileMoneyTransactionData,
        num_data_points=num_data_points,
        keep_optional_columns=keep_optional_columns,
    )

    corrected_mobile_money_df = (
        correct_generated_synthetic_mobile_money_transaction_data(
            synthetic_mobile_money_df
        )
    )

    # Validate the corrected DataFrame against the MobileMoneyTransactionData schema and num datapoints
    assert len(corrected_mobile_money_df) == num_data_points
    assert (
        validate_dataframe(
            corrected_mobile_money_df,
            required_schema=MobileMoneyTransactionData,
            check_data_points=True,
        )
        is None
    )


def test_correct_generated_synthetic_cdr_data():
    num_data_points = 100
    num_unique_antenna_ids = 5
    synthetic_cdr_df = generate_synthetic_data(
        schema=CallDataRecordData,
        num_data_points=num_data_points,
        keep_optional_columns=True,
    )

    corrected_cdr_df = correct_generated_synthetic_cdr_data(
        synthetic_cdr_df,
        num_unique_antenna_ids,
    )

    # Validate the corrected DataFrame against the CallDataRecordData schema and num datapoints
    assert len(corrected_cdr_df) == num_data_points
    assert corrected_cdr_df.caller_antenna_id.nunique() <= num_unique_antenna_ids
    assert corrected_cdr_df.recipient_antenna_id.nunique() <= num_unique_antenna_ids
    assert np.all(
        corrected_cdr_df.loc[corrected_cdr_df.transaction_type == "text", "duration"]
        == 0
    )
    assert (
        validate_dataframe(
            corrected_cdr_df,
            required_schema=CallDataRecordData,
            check_data_points=True,
        )
        is None
    )


@pytest.mark.parametrize(
    "num_regions",
    [1, 2],
)
def test_generate_synthetic_shapefile(num_regions):
    antenna_df = pd.DataFrame(ANTENNA_DATA).copy()
    shapefile_gdf = generate_synthetic_shapefile(
        antenna_df=antenna_df, num_regions=num_regions
    )

    # Validate the synthetic shapefile GeoDataFrame against the expected number of regions
    assert len(shapefile_gdf) == num_regions
