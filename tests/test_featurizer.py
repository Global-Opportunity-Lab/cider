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


from conftest import (
    CDR_DATA,
    MOBILE_DATA_USAGE_DATA,
    MOBILE_MONEY_TRANSACTION_DATA,
    RECHARGE_DATA,
    ANTENNA_DATA,
)
import numpy as np
import pandas as pd
import deepdiff
import pytest
from cider.schemas import CallDataRecordData
from cider.featurizer.schemas import (
    CallDataRecordTagged,
    MobileMoneyDataWithDirection,
    MobileMoneyDataWithDay,
    MobileDataUsageDataWithDay,
    RechargeDataWithDay,
    AntennaDataGeometry,
    AntennaDataGeometryWithRegion,
)
from cider.featurizer.dependencies import (
    filter_to_datetime,
    get_spammers_from_cdr_data,
    get_outlier_days_from_cdr_data,
    get_static_diagnostic_statistics,
    get_timeseries_diagnostic_statistics,
    identify_daytime,
    identify_weekend,
    swap_caller_and_recipient,
    identify_and_tag_conversations,
    identify_mobile_money_transaction_direction,
)
from cider.featurizer.core import (
    get_active_days,
    get_number_of_contacts_per_caller,
    get_call_duration_stats,
    get_percentage_of_nocturnal_interactions,
    get_percentage_of_initiated_conversations,
    get_percentage_of_initiated_calls,
    get_text_response_time_delay_stats,
    get_text_response_rate,
    get_entropy_of_interactions_per_caller,
    get_outgoing_interaction_fraction_stats,
    get_interaction_stats_per_caller,
    get_inter_event_time_stats,
    get_pareto_principle_interaction_stats,
    get_pareto_principle_call_duration_stats,
    get_number_of_interactions_per_user,
    get_number_of_antennas,
    get_entropy_of_antennas_per_caller,
    get_pareto_principle_antennas,
    get_average_num_of_interactions_from_home_antennas,
    get_international_interaction_statistics,
    get_radius_of_gyration,
    get_mobile_data_stats,
    get_mobile_money_amount_stats,
    get_mobile_money_transaction_stats,
    get_mobile_money_balance_stats,
    get_recharge_amount_stats,
    get_caller_counts_per_region,
)


class TestFeaturizerDependencies:

    @pytest.mark.parametrize(
        "dataset,expected_rows",
        [
            (CDR_DATA, 2),
            (MOBILE_DATA_USAGE_DATA, 4),
            (MOBILE_MONEY_TRANSACTION_DATA, 4),
            (RECHARGE_DATA, 4),
        ],
    )
    def test_filter_to_datetime(self, dataset, expected_rows):
        df = pd.DataFrame(dataset)
        filtered_data = filter_to_datetime(
            df,
            filter_start_date=pd.to_datetime("2023-01-02"),
            filter_end_date=pd.to_datetime("2023-01-03"),
        )

        assert all(
            (filtered_data["timestamp"] >= pd.to_datetime("2023-01-02 00:00:00"))
            & (filtered_data["timestamp"] <= pd.to_datetime("2023-01-03 23:59:59"))
        )
        assert len(filtered_data) == expected_rows

        df.pop("timestamp")
        with pytest.raises(
            ValueError, match="Dataframe must contain 'timestamp' column"
        ):
            filter_to_datetime(
                df,
                filter_start_date=pd.to_datetime("2023-01-02"),
                filter_end_date=pd.to_datetime("2023-01-03"),
            )

    def test_get_spammers_from_cdr_data(self):
        cdr = pd.DataFrame(CDR_DATA)
        # Add spammer data
        spammer_data = {
            "caller_id": ["spammer_1"] * 300,
            "recipient_id": ["recipient_spam"] * 300,
            "caller_antenna_id": ["antenna_spam"] * 300,
            "timestamp": pd.date_range(
                start="2023-01-01 00:00:00", periods=300, freq="5min"
            ),
            "duration": [60] * 300,
            "transaction_type": ["call"] * 300,
            "transaction_scope": ["domestic"] * 300,
        }
        spammer_cdr = pd.DataFrame(spammer_data)
        cdr_with_spammer = pd.concat([cdr, spammer_cdr], ignore_index=True)

        spammer_ids = get_spammers_from_cdr_data(
            cdr_with_spammer, threshold_of_calls_per_day=100
        )

        assert spammer_ids == ["spammer_1"]

        for col in [
            key
            for key, field in CallDataRecordData.model_fields.items()
            if field.is_required()
        ]:
            cdr_with_spammer_missing = cdr_with_spammer.copy()
            cdr_with_spammer_missing.rename(columns={col: "wrong_column"}, inplace=True)

            with pytest.raises(
                ValueError,
                match=f"The following required columns are missing from the dataframe: {set([col])}",
            ):
                get_spammers_from_cdr_data(
                    cdr_with_spammer_missing, threshold_of_calls_per_day=100
                )

    def test_get_outlier_days_from_cdr_data(self):
        cdr = pd.DataFrame(CDR_DATA)
        # Add outlier day data
        outlier_data = {
            "caller_id": ["caller_outlier"] * 1000,
            "recipient_id": ["recipient_outlier"] * 1000,
            "caller_antenna_id": ["antenna_outlier"] * 1000,
            "timestamp": pd.date_range(
                start="2023-01-10 00:00:01", periods=1000, freq="s"
            ),
            "duration": [60] * 1000,
            "transaction_type": ["call"] * 1000,
            "transaction_scope": ["domestic"] * 1000,
        }
        outlier_cdr = pd.DataFrame(outlier_data)
        cdr_with_outlier = pd.concat([cdr, outlier_cdr], ignore_index=True)

        outlier_days = get_outlier_days_from_cdr_data(
            cdr_with_outlier, zscore_threshold=1.0
        )

        assert pd.to_datetime("2023-01-10").date() in outlier_days
        assert len(outlier_days) == 1

        for col in [
            key
            for key, field in CallDataRecordData.model_fields.items()
            if field.is_required()
        ]:
            cdr_with_outlier_missing = cdr_with_outlier.copy()
            cdr_with_outlier_missing.rename(columns={col: "wrong_column"}, inplace=True)

            with pytest.raises(
                ValueError,
                match=f"The following required columns are missing from the dataframe: {set([col])}",
            ):
                get_outlier_days_from_cdr_data(
                    cdr_with_outlier_missing, zscore_threshold=1.0
                )

    @pytest.mark.parametrize(
        "data",
        [
            CDR_DATA,
            MOBILE_DATA_USAGE_DATA,
            MOBILE_MONEY_TRANSACTION_DATA,
            RECHARGE_DATA,
        ],
    )
    def test_get_static_diagnostic_statistics(self, data):
        df = pd.DataFrame(data)
        stats = get_static_diagnostic_statistics(df)

        assert stats.total_transactions == len(df)
        assert stats.num_unique_callers == df["caller_id"].nunique()
        assert (
            stats.num_days == (df["timestamp"].max() - df["timestamp"].min()).days + 1
        )
        if "recipient_id" in df.columns:
            assert stats.num_unique_recipients == df["recipient_id"].nunique()
        else:
            assert stats.num_unique_recipients == 0

        for col in ["caller_id", "timestamp"]:
            df_missing = df.drop(columns=[col])
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id' and 'timestamp' columns",
            ):
                get_static_diagnostic_statistics(df_missing)

    @pytest.mark.parametrize(
        "data",
        [
            CDR_DATA,
            MOBILE_DATA_USAGE_DATA,
            MOBILE_MONEY_TRANSACTION_DATA,
            RECHARGE_DATA,
        ],
    )
    def test_get_timeseries_diagnostic_statistics(self, data):
        df = pd.DataFrame(data)

        static_data = get_static_diagnostic_statistics(df)

        unique_days = df["timestamp"].dt.date.nunique()
        timeseries_stats = get_timeseries_diagnostic_statistics(df)
        assert set(timeseries_stats.columns).issubset(
            {
                "day",
                "transaction_type",
                "total_transactions",
                "num_unique_callers",
                "num_unique_recipients",
            }
        )
        assert timeseries_stats.day.nunique() == unique_days
        assert (
            static_data.total_transactions
            == timeseries_stats["total_transactions"].sum()
        )

        for col in ["caller_id", "timestamp"]:
            df_missing = df.drop(columns=[col])
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id' and 'timestamp' columns",
            ):
                get_timeseries_diagnostic_statistics(df_missing)

    def test_identify_daytime(self, spark):
        spark_cdr_data = spark.createDataFrame(pd.DataFrame(CDR_DATA))
        cdr_spark_with_daytime = identify_daytime(
            spark_cdr_data, day_start=12, day_end=17
        )
        cdr_with_daytime = cdr_spark_with_daytime.toPandas()

        assert "is_daytime" in cdr_with_daytime.columns
        assert cdr_with_daytime.is_daytime.values.tolist() == [0, 1, 1, 1, 0, 0]

        pd_cdr_data = pd.DataFrame(CDR_DATA).drop(columns=["timestamp"])
        spark_cdr_data_no_timestamp = spark.createDataFrame(pd_cdr_data)
        with pytest.raises(
            ValueError, match="Dataframe must contain 'timestamp' column"
        ):
            identify_daytime(spark_cdr_data_no_timestamp)

    def test_identify_weekend(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)
        pd_cdr_data["day"] = pd_cdr_data["timestamp"].dt.date
        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        cdr_spark_with_weekend = identify_weekend(spark_cdr_data, weekend_days=[2, 6])
        cdr_with_weekend = cdr_spark_with_weekend.toPandas()

        assert "is_weekend" in cdr_with_weekend.columns
        assert cdr_with_weekend.is_weekend.values.tolist() == [0, 1, 1, 0, 0, 1]

        pd_cdr_data = pd.DataFrame(CDR_DATA).drop(columns=["timestamp"])
        spark_cdr_data_no_timestamp = spark.createDataFrame(pd_cdr_data)
        with pytest.raises(
            ValueError, match="Dataframe must contain 'timestamp' column"
        ):
            identify_weekend(spark_cdr_data_no_timestamp)

    def test_swap_caller_and_recipient(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)
        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_swapped = swap_caller_and_recipient(spark_cdr_data)
        pd_cdr_swapped = spark_cdr_swapped.toPandas()

        assert len(pd_cdr_swapped) == 2 * len(pd_cdr_data)
        assert set(pd_cdr_swapped.caller_id.unique()) == set(
            pd_cdr_swapped.recipient_id.unique()
        )
        assert "direction_of_transaction" in pd_cdr_swapped.columns
        assert set(pd_cdr_swapped.direction_of_transaction.unique()) == {
            "outgoing",
            "incoming",
        }

        for col in [
            key
            for key, field in CallDataRecordData.model_fields.items()
            if field.is_required()
        ]:
            pd_cdr_data_missing = pd_cdr_data.drop(columns=[col])
            spark_cdr_missing = spark.createDataFrame(pd_cdr_data_missing)

            with pytest.raises(
                ValueError,
                match=f"The following required columns are missing from the dataframe: {set([col])}",
            ):
                swap_caller_and_recipient(spark_cdr_missing)

    def test_identify_and_tag_conversations(self, spark):
        conversations = {
            "caller_id": ["user_1"] * 6,
            "recipient_id": ["user_2"] * 6,
            "timestamp": pd.to_datetime(
                [
                    "2023-01-10 10:00:00",
                    "2023-01-10 10:30:00",
                    "2023-01-10 10:45:00",
                    "2023-01-11 13:10:00",
                    "2023-01-11 13:30:00",
                    "2023-01-11 13:55:00",
                ]
            ),
            "duration": [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
            "transaction_scope": ["domestic"] * 6,
            "transaction_type": ["text", "text", "call", "text", "text", "text"],
        }
        pd_cdr_data = pd.concat(
            [pd.DataFrame(CDR_DATA).copy(), pd.DataFrame(conversations)],
            ignore_index=True,
        )
        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_tagged = identify_and_tag_conversations(spark_cdr_data, max_wait=3600)
        pd_cdr_tagged = spark_cdr_tagged.toPandas()

        assert "conversation" in pd_cdr_tagged.columns
        convo_times = pd_cdr_tagged["conversation"].dropna().unique()
        assert len(convo_times) == 5

        for col in [
            key
            for key, field in CallDataRecordData.model_fields.items()
            if field.is_required()
        ]:
            pd_cdr_data_missing = pd_cdr_data.drop(columns=[col])
            spark_cdr_missing = spark.createDataFrame(pd_cdr_data_missing)

            with pytest.raises(
                ValueError,
                match=f"The following required columns are missing from the dataframe: {set([col])}",
            ):
                identify_and_tag_conversations(spark_cdr_missing)

    def test_identify_mobile_money_transaction_direction(self, spark):
        pd_mobile_money_data = pd.DataFrame(MOBILE_MONEY_TRANSACTION_DATA)
        pd_mobile_money_data["day"] = pd_mobile_money_data["timestamp"].dt.date
        spark_mobile_money_data = spark.createDataFrame(pd_mobile_money_data)
        spark_mobile_money_direction = identify_mobile_money_transaction_direction(
            spark_mobile_money_data
        )

        assert set(spark_mobile_money_direction.toPandas().columns) == set(
            MobileMoneyDataWithDirection.model_fields.keys()
        )

        for col in [
            key
            for key, field in MobileMoneyDataWithDay.model_fields.items()
            if field.is_required()
        ]:
            pd_mobile_money_data_missing = pd_mobile_money_data.drop(columns=[col])
            spark_mobile_money_missing = spark.createDataFrame(
                pd_mobile_money_data_missing
            )

            with pytest.raises(
                ValueError,
                match=f"The following required columns are missing from the dataframe: {set([col])}",
            ):
                identify_mobile_money_transaction_direction(spark_mobile_money_missing)


class TestFeaturizerCoreCDRData:

    FUNCTION_MAP = {
        "get_active_days": get_active_days,
        "get_number_of_contacts_per_caller": get_number_of_contacts_per_caller,
        "get_call_duration_stats": get_call_duration_stats,
        "get_percentage_of_nocturnal_interactions": get_percentage_of_nocturnal_interactions,
        "get_percentage_of_initiated_conversations": get_percentage_of_initiated_conversations,
        "get_percentage_of_initiated_calls": get_percentage_of_initiated_calls,
        "get_text_response_time_delay_stats": get_text_response_time_delay_stats,
        "get_text_response_rate": get_text_response_rate,
        "get_entropy_of_interactions_per_caller": get_entropy_of_interactions_per_caller,
        "get_outgoing_interaction_fraction_stats": get_outgoing_interaction_fraction_stats,
        "get_interaction_stats_per_caller": get_interaction_stats_per_caller,
        "get_inter_event_time_stats": get_inter_event_time_stats,
        "get_pareto_principle_interaction_stats": get_pareto_principle_interaction_stats,
        "get_pareto_principle_call_duration_stats": get_pareto_principle_call_duration_stats,
        "get_number_of_interactions_per_user": get_number_of_interactions_per_user,
        "get_number_of_antennas": get_number_of_antennas,
        "get_entropy_of_antennas_per_caller": get_entropy_of_antennas_per_caller,
        "get_pareto_principle_antennas": get_pareto_principle_antennas,
        "get_average_num_of_interactions_from_home_antennas": get_average_num_of_interactions_from_home_antennas,
        "get_international_interaction_statistics": get_international_interaction_statistics,
    }

    @pytest.fixture
    def spark_cdr_with_conversations(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)
        pd_cdr_data.loc[:, "day"] = pd_cdr_data["timestamp"].dt.date

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_cdr_swapped = swap_caller_and_recipient(spark_cdr_with_weekend)
        spark_cdr_with_conversations = identify_and_tag_conversations(spark_cdr_swapped)

        return spark_cdr_with_conversations

    def _get_expected_results(self, function: str) -> pd.DataFrame:
        expected_results = {}
        match function:
            case "get_active_days":
                expected_results = {
                    "active_days_all": [2.0, 2.0, 5.0, 2.0],
                    "active_days_weekday": [2.0, 1.0, 4.0, 2.0],
                    "active_days_weekend": [0.0, 1.0, 1.0, 0.0],
                    "active_days_day": [2.0, 2.0, 4.0, 1.0],
                    "active_days_night": [0.0, 0.0, 1.0, 1.0],
                    "active_days_weekday_day": [2.0, 1.0, 3.0, 1.0],
                    "active_days_weekday_night": [0.0, 0.0, 1.0, 1.0],
                    "active_days_weekend_day": [0.0, 1.0, 1.0, 0.0],
                    "active_days_weekend_night": [0.0, 0.0, 0.0, 0.0],
                }
            case "get_number_of_contacts_per_caller":
                expected_results = {
                    "weekday_nighttime_text_num_unique_contacts": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_text_num_unique_contacts": [0.0, 0.0, 0.0, 0.0],
                    "weekday_nighttime_call_num_unique_contacts": [0.0, 0.0, 1.0, 1.0],
                    "weekday_daytime_call_num_unique_contacts": [1.0, 1.0, 0.0, 0.0],
                    "weekend_nighttime_text_num_unique_contacts": [0.0, 0.0, 0.0, 0.0],
                    "weekend_daytime_text_num_unique_contacts": [0.0, 0.0, 0.0, 0.0],
                    "weekend_nighttime_call_num_unique_contacts": [0.0, 0.0, 0.0, 0.0],
                    "weekend_daytime_call_num_unique_contacts": [0.0, 0.0, 0.0, 0.0],
                    "nighttime_text_num_unique_contacts": [0.0, 0.0, 0.0, 0.0],
                    "daytime_text_num_unique_contacts": [0.0, 0.0, 0.0, 0.0],
                    "weekend_text_num_unique_contacts": [0.0, 0.0, 0.0, 0.0],
                    "weekday_text_num_unique_contacts": [0.0, 0.0, 0.0, 0.0],
                    "nighttime_call_num_unique_contacts": [0.0, 0.0, 0.0, 0.0],
                    "daytime_call_num_unique_contacts": [0.0, 0.0, 0.0, 0.0],
                    "weekend_call_num_unique_contacts": [0.0, 0.0, 0.0, 0.0],
                    "weekday_call_num_unique_contacts": [0.0, 0.0, 0.0, 0.0],
                    "call_num_unique_contacts": [1.0, 1.0, 1.0, 1.0],
                    "text_num_unique_contacts": [0.0, 0.0, 0.0, 0.0],
                }
            case "get_call_duration_stats":
                expected_results = {
                    "weekday_nighttime_mean_duration": [0.0, 0.0, 150.0, 150.0],
                    "weekend_nighttime_mean_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_mean_duration": [100.0, 200.0, 0.0, 0.0],
                    "weekend_daytime_mean_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekday_nighttime_median_duration": [0.0, 0.0, 150.0, 150.0],
                    "weekend_nighttime_median_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_median_duration": [100.0, 200.0, 0.0, 0.0],
                    "weekend_daytime_median_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekday_nighttime_max_duration": [0.0, 0.0, 150.0, 150.0],
                    "weekend_nighttime_max_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_max_duration": [100.0, 200.0, 0.0, 0.0],
                    "weekend_daytime_max_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekday_nighttime_min_duration": [0.0, 0.0, 150.0, 150.0],
                    "weekend_nighttime_min_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_min_duration": [100.0, 200.0, 0.0, 0.0],
                    "weekend_daytime_min_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekday_nighttime_std_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekend_nighttime_std_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_std_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekend_daytime_std_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekday_nighttime_skewness_duration": [
                        0.0,
                        0.0,
                        np.nan,
                        np.nan,
                    ],
                    "weekend_nighttime_skewness_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_skewness_duration": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_skewness_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekday_nighttime_kurtosis_duration": [
                        0.0,
                        0.0,
                        np.nan,
                        np.nan,
                    ],
                    "weekend_nighttime_kurtosis_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_kurtosis_duration": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_kurtosis_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekend_mean_duration": [0.0, 0.0, 75.0, 75.0],
                    "weekday_mean_duration": [100.0, 200.0, 75.0, 75.0],
                    "weekend_median_duration": [0.0, 0.0, 75.0, 75.0],
                    "weekday_median_duration": [100.0, 200.0, 75.0, 75.0],
                    "weekend_max_duration": [0.0, 0.0, 75.0, 75.0],
                    "weekday_max_duration": [100.0, 200.0, 150.0, 150.0],
                    "weekend_min_duration": [0.0, 0.0, 75.0, 75.0],
                    "weekday_min_duration": [100.0, 200.0, 150.0, 150.0],
                    "weekend_std_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekday_std_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekend_skewness_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekday_skewness_duration": [0.0, 0.0, np.nan, np.nan],
                    "weekend_kurtosis_duration": [0.0, 0.0, 0.0, 0.0],
                    "weekday_kurtosis_duration": [0.0, 0.0, np.nan, np.nan],
                    "daytime_mean_duration": [100.0, 200.0, 0.0, 0.0],
                    "nighttime_mean_duration": [0.0, 0.0, 150.0, 150.0],
                    "daytime_median_duration": [100.0, 200.0, 0.0, 0.0],
                    "nighttime_median_duration": [0.0, 0.0, 150.0, 150.0],
                    "daytime_max_duration": [100.0, 200.0, 0.0, 0.0],
                    "nighttime_max_duration": [0.0, 0.0, 150.0, 150.0],
                    "daytime_min_duration": [100.0, 200.0, 0.0, 0.0],
                    "nighttime_min_duration": [0.0, 0.0, 150.0, 150.0],
                    "daytime_std_duration": [0.0, 0.0, 0.0, 0.0],
                    "nighttime_std_duration": [0.0, 0.0, 0.0, 0.0],
                    "daytime_skewness_duration": [np.nan, np.nan, 0.0, 0.0],
                    "nighttime_skewness_duration": [0.0, 0.0, np.nan, np.nan],
                    "daytime_kurtosis_duration": [np.nan, np.nan, 0.0, 0.0],
                    "nighttime_kurtosis_duration": [0.0, 0.0, np.nan, np.nan],
                    "mean_duration": [50.0, 100.0, 75.0, 75.0],
                    "median_duration": [0.0, 0.0, 75.0, 75.0],
                    "max_duration": [100.0, 200.0, 150.0, 150.0],
                    "min_duration": [0.0, 0.0, 75.0, 75.0],
                    "std_duration": [0.0, 0.0, 0.0, 0.0],
                    "skewness_duration": [0.0, 0.0, np.nan, np.nan],
                    "kurtosis_duration": [0.0, 0.0, np.nan, np.nan],
                }
            case "get_percentage_of_nocturnal_interactions":
                expected_results = {
                    "weekday_text_percentage_nocturnal_interactions": [
                        0.0,
                        0.0,
                        50.0,
                        0.0,
                    ],
                    "weekend_text_percentage_nocturnal_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        16.667,
                    ],
                    "weekday_call_percentage_nocturnal_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_call_percentage_nocturnal_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_percentage_nocturnal_interactions": [
                        0.0,
                        0.0,
                        33.333,
                        8.333,
                    ],
                    "call_percentage_nocturnal_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                }
            case "get_percentage_of_initiated_conversations":
                expected_results = {
                    "weekday_nighttime_percentage_initiated_conversations": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_nighttime_percentage_initiated_conversations": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_percentage_initiated_conversations": [
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "weekend_daytime_percentage_initiated_conversations": [
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    "nighttime_percentage_initiated_conversations": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "daytime_percentage_initiated_conversations": [
                        1.0,
                        0.5,
                        0.0,
                        1.0,
                    ],
                    "weekday_percentage_initiated_conversations": [
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "weekend_percentage_initiated_conversations": [
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    "percentage_initiated_conversations": [
                        1.0,
                        0.25,
                        0.0,
                        1.0,
                    ],
                }
            case "get_percentage_of_initiated_calls":
                expected_results = {
                    "weekday_nighttime_percentage_initiated_calls": [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "weekend_nighttime_percentage_initiated_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_percentage_initiated_calls": [1.0, 1.0, 0.0, 0.0],
                    "weekend_daytime_percentage_initiated_calls": [0.0, 0.0, 0.0, 0.0],
                    "nighttime_percentage_initiated_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                    ],
                    "daytime_percentage_initiated_calls": [1.0, 1.0, 0.0, 0.0],
                    "weekday_percentage_initiated_calls": [1.0, 1.0, 0.0, 0.0],
                    "weekend_percentage_initiated_calls": [0.0, 0.0, 0.0, 0.0],
                    "percentage_initiated_calls": [1.0, 1.0, 0.0, 0.25],
                }
            case "get_text_response_time_delay_stats":
                expected_results = {
                    "weekday_nighttime_mean_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekend_nighttime_mean_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_mean_response_time_delay": [
                        np.nan,
                        0.0,
                        0.0,
                        np.nan,
                    ],
                    "weekend_daytime_mean_response_time_delay": [
                        0.0,
                        np.nan,
                        np.nan,
                        0.0,
                    ],
                    "weekday_nighttime_median_response_time_delay": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_nighttime_median_response_time_delay": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_median_response_time_delay": [
                        np.nan,
                        0.0,
                        0.0,
                        np.nan,
                    ],
                    "weekend_daytime_median_response_time_delay": [
                        0.0,
                        np.nan,
                        np.nan,
                        0.0,
                    ],
                    "weekday_nighttime_max_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekend_nighttime_max_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_max_response_time_delay": [
                        np.nan,
                        0.0,
                        0.0,
                        np.nan,
                    ],
                    "weekend_daytime_max_response_time_delay": [
                        0.0,
                        np.nan,
                        np.nan,
                        0.0,
                    ],
                    "weekday_nighttime_min_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekend_nighttime_min_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_min_response_time_delay": [
                        np.nan,
                        0.0,
                        0.0,
                        np.nan,
                    ],
                    "weekend_daytime_min_response_time_delay": [
                        0.0,
                        np.nan,
                        np.nan,
                        0.0,
                    ],
                    "weekday_nighttime_std_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekend_nighttime_std_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_std_response_time_delay": [
                        np.nan,
                        0.0,
                        0.0,
                        np.nan,
                    ],
                    "weekend_daytime_std_response_time_delay": [
                        0.0,
                        np.nan,
                        np.nan,
                        0.0,
                    ],
                    "weekday_nighttime_skewness_response_time_delay": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_nighttime_skewness_response_time_delay": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_skewness_response_time_delay": [
                        np.nan,
                        0.0,
                        0.0,
                        np.nan,
                    ],
                    "weekend_daytime_skewness_response_time_delay": [
                        0.0,
                        np.nan,
                        np.nan,
                        0.0,
                    ],
                    "weekday_nighttime_kurtosis_response_time_delay": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_nighttime_kurtosis_response_time_delay": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_kurtosis_response_time_delay": [
                        np.nan,
                        0.0,
                        0.0,
                        np.nan,
                    ],
                    "weekend_daytime_kurtosis_response_time_delay": [
                        0.0,
                        np.nan,
                        np.nan,
                        0.0,
                    ],
                    "weekday_mean_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekday_std_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekday_median_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekday_min_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekday_max_response_time_delay": [np.nan, 0.0, 0.0, np.nan],
                    "weekday_skewness_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekday_kurtosis_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekend_mean_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekend_std_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekend_median_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekend_min_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekend_max_response_time_delay": [0.0, np.nan, np.nan, 0.0],
                    "weekend_skewness_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "weekend_kurtosis_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "daytime_mean_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "daytime_std_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "daytime_median_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "daytime_min_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "daytime_max_response_time_delay": [np.nan, 0.0, 0.0, np.nan],
                    "daytime_skewness_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "daytime_kurtosis_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "nighttime_mean_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "nighttime_std_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "nighttime_median_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "nighttime_min_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "nighttime_max_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "nighttime_skewness_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "nighttime_kurtosis_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "mean_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "std_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "median_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "min_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "max_response_time_delay": [np.nan, 0.0, 0.0, np.nan],
                    "skewness_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                    "kurtosis_response_time_delay": [0.0, 0.0, 0.0, 0.0],
                }
            case "get_text_response_rate":
                expected_results = {
                    "weekday_nighttime_text_response_rate": [0.0],
                    "weekend_nighttime_text_response_rate": [0.0],
                    "weekday_daytime_text_response_rate": [0.0],
                    "weekend_daytime_text_response_rate": [0.0],
                    "weekday_text_response_rate": [0.0],
                    "weekend_text_response_rate": [0.0],
                    "text_response_rate": [0.0],
                    "daytime_text_response_rate": [0.0],
                    "nighttime_text_response_rate": [0.0],
                }

            case "get_entropy_of_interactions_per_caller":
                expected_results = {
                    "weekday_nighttime_text_entropy_of_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_nighttime_call_entropy_of_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_nighttime_text_entropy_of_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_nighttime_call_entropy_of_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_text_entropy_of_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_call_entropy_of_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_text_entropy_of_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_call_entropy_of_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_text_entropy_of_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_text_entropy_of_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "nighttime_text_entropy_of_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "daytime_text_entropy_of_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_call_entropy_of_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_call_entropy_of_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "nighttime_call_entropy_of_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "daytime_call_entropy_of_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_entropy_of_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_entropy_of_interactions": [0.0, 0.0, 0.0, 0.0],
                }

            case "get_outgoing_interaction_fraction_stats":
                expected_results = {
                    # Both dimensions (transaction_type + is_weekend + is_daytime)
                    "text_weekday_daytime_mean_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_daytime_mean_fraction_of_outgoing_calls": [
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_nighttime_mean_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_nighttime_mean_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "text_weekend_daytime_mean_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_daytime_mean_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_nighttime_mean_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_nighttime_mean_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_daytime_min_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_daytime_min_fraction_of_outgoing_calls": [
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_nighttime_min_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_nighttime_min_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "text_weekend_daytime_min_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_daytime_min_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_nighttime_min_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_nighttime_min_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_daytime_max_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_daytime_max_fraction_of_outgoing_calls": [
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_nighttime_max_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_nighttime_max_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "text_weekend_daytime_max_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_daytime_max_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_nighttime_max_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_nighttime_max_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_daytime_std_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_daytime_std_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_nighttime_std_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_nighttime_std_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_daytime_std_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_daytime_std_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_nighttime_std_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_nighttime_std_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_daytime_median_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_daytime_median_fraction_of_outgoing_calls": [
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_nighttime_median_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_nighttime_median_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "text_weekend_daytime_median_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_daytime_median_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_nighttime_median_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_nighttime_median_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_daytime_skewness_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_daytime_skewness_fraction_of_outgoing_calls": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_nighttime_skewness_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_nighttime_skewness_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        np.nan,
                    ],
                    "text_weekend_daytime_skewness_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_daytime_skewness_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_nighttime_skewness_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_nighttime_skewness_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_daytime_kurtosis_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_daytime_kurtosis_fraction_of_outgoing_calls": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_nighttime_kurtosis_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_nighttime_kurtosis_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        np.nan,
                    ],
                    "text_weekend_daytime_kurtosis_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_daytime_kurtosis_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_nighttime_kurtosis_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_nighttime_kurtosis_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    # Only is_weekend dimension
                    "text_weekday_mean_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_mean_fraction_of_outgoing_calls": [
                        1.0,
                        1.0,
                        0.0,
                        0.5,
                    ],
                    "text_weekend_mean_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_mean_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_min_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_min_fraction_of_outgoing_calls": [1.0, 1.0, 0.0, 0.5],
                    "text_weekend_min_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_min_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "text_weekday_max_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_max_fraction_of_outgoing_calls": [1.0, 1.0, 0.0, 1.0],
                    "text_weekend_max_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_max_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "text_weekday_std_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_std_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.5],
                    "text_weekend_std_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_std_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "text_weekday_median_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_median_fraction_of_outgoing_calls": [
                        1.0,
                        1.0,
                        0.0,
                        0.5,
                    ],
                    "text_weekend_median_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_median_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_skewness_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_skewness_fraction_of_outgoing_calls": [
                        np.nan,
                        np.nan,
                        0.0,
                        np.nan,
                    ],
                    "text_weekend_skewness_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_skewness_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_kurtosis_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_kurtosis_fraction_of_outgoing_calls": [
                        np.nan,
                        np.nan,
                        0.0,
                        np.nan,
                    ],
                    "text_weekend_kurtosis_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_kurtosis_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    # Only is_daytime dimension
                    "text_daytime_mean_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_daytime_mean_fraction_of_outgoing_calls": [
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    "text_nighttime_mean_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_nighttime_mean_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "text_daytime_min_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "call_daytime_min_fraction_of_outgoing_calls": [1.0, 1.0, 0.0, 0.0],
                    "text_nighttime_min_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_nighttime_min_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "text_daytime_max_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "call_daytime_max_fraction_of_outgoing_calls": [1.0, 1.0, 0.0, 0.0],
                    "text_nighttime_max_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_nighttime_max_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "text_daytime_std_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "call_daytime_std_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "text_nighttime_std_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_nighttime_std_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_daytime_median_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_daytime_median_fraction_of_outgoing_calls": [
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    "text_nighttime_median_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_nighttime_median_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "text_daytime_skewness_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_daytime_skewness_fraction_of_outgoing_calls": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    "text_nighttime_skewness_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_nighttime_skewness_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        np.nan,
                    ],
                    "text_daytime_kurtosis_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_daytime_kurtosis_fraction_of_outgoing_calls": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    "text_nighttime_kurtosis_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_nighttime_kurtosis_fraction_of_outgoing_calls": [
                        0.0,
                        0.0,
                        0.0,
                        np.nan,
                    ],
                    # No time dimensions (transaction_type only)
                    "text_mean_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "call_mean_fraction_of_outgoing_calls": [1.0, 1.0, 0.0, 0.5],
                    "text_min_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "call_min_fraction_of_outgoing_calls": [1.0, 1.0, 0.0, 0.5],
                    "text_max_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "call_max_fraction_of_outgoing_calls": [1.0, 1.0, 0.0, 1.0],
                    "text_std_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "call_std_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.5],
                    "text_median_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "call_median_fraction_of_outgoing_calls": [1.0, 1.0, 0.0, 0.5],
                    "text_skewness_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "call_skewness_fraction_of_outgoing_calls": [
                        np.nan,
                        np.nan,
                        0.0,
                        np.nan,
                    ],
                    "text_kurtosis_fraction_of_outgoing_calls": [0.0, 0.0, 0.0, 0.0],
                    "call_kurtosis_fraction_of_outgoing_calls": [
                        np.nan,
                        np.nan,
                        0.0,
                        np.nan,
                    ],
                }
            case "get_interaction_stats_per_caller":
                expected_results = {
                    # No time dimensions (allweek_allday)
                    "text_mean_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_mean_interaction_count": [1.0, 1.0, 0.5, 0.5],
                    "text_min_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_min_interaction_count": [1.0, 1.0, 0.0, 0.0],
                    "text_max_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_max_interaction_count": [1.0, 1.0, 1.0, 1.0],
                    "text_std_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_std_interaction_count": [0.0, 0.0, 0.5, 0.5],
                    "text_median_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_median_interaction_count": [1.0, 1.0, 0.5, 0.5],
                    "text_skewness_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_skewness_interaction_count": [np.nan, np.nan, np.nan, np.nan],
                    "text_kurtosis_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_kurtosis_interaction_count": [np.nan, np.nan, np.nan, np.nan],
                    # Only daytime dimension (allweek_day)
                    "text_daytime_mean_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_daytime_mean_interaction_count": [1.0, 1.0, 0.0, 0.0],
                    "text_daytime_min_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_daytime_min_interaction_count": [1.0, 1.0, 0.0, 0.0],
                    "text_daytime_max_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_daytime_max_interaction_count": [1.0, 1.0, 0.0, 0.0],
                    "text_daytime_std_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_daytime_std_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "text_daytime_median_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_daytime_median_interaction_count": [1.0, 1.0, 0.0, 0.0],
                    "text_daytime_skewness_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_daytime_skewness_interaction_count": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    "text_daytime_kurtosis_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_daytime_kurtosis_interaction_count": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    # Only nighttime dimension (allweek_night)
                    "text_nighttime_mean_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_nighttime_mean_interaction_count": [0.0, 0.0, 1.0, 1.0],
                    "text_nighttime_min_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_nighttime_min_interaction_count": [0.0, 0.0, 1.0, 1.0],
                    "text_nighttime_max_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_nighttime_max_interaction_count": [0.0, 0.0, 1.0, 1.0],
                    "text_nighttime_std_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_nighttime_std_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "text_nighttime_median_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_nighttime_median_interaction_count": [0.0, 0.0, 1.0, 1.0],
                    "text_nighttime_skewness_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_nighttime_skewness_interaction_count": [
                        0.0,
                        0.0,
                        np.nan,
                        np.nan,
                    ],
                    "text_nighttime_kurtosis_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_nighttime_kurtosis_interaction_count": [
                        0.0,
                        0.0,
                        np.nan,
                        np.nan,
                    ],
                    # Only weekday dimension (weekday_allday)
                    "text_weekday_mean_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_mean_interaction_count": [1.0, 1.0, 0.5, 0.5],
                    "text_weekday_min_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_min_interaction_count": [1.0, 1.0, 0.0, 0.0],
                    "text_weekday_max_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_max_interaction_count": [1.0, 1.0, 1.0, 1.0],
                    "text_weekday_std_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_std_interaction_count": [0.0, 0.0, 0.5, 0.5],
                    "text_weekday_median_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_median_interaction_count": [1.0, 1.0, 0.5, 0.5],
                    "text_weekday_skewness_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_skewness_interaction_count": [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "text_weekday_kurtosis_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_kurtosis_interaction_count": [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    # Weekday + daytime
                    "text_weekday_daytime_mean_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_daytime_mean_interaction_count": [1.0, 1.0, 0.0, 0.0],
                    "text_weekday_daytime_min_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_daytime_min_interaction_count": [1.0, 1.0, 0.0, 0.0],
                    "text_weekday_daytime_max_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_daytime_max_interaction_count": [1.0, 1.0, 0.0, 0.0],
                    "text_weekday_daytime_std_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_daytime_std_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "text_weekday_daytime_median_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_daytime_median_interaction_count": [
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_daytime_skewness_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_daytime_skewness_interaction_count": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_daytime_kurtosis_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_daytime_kurtosis_interaction_count": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    # Weekday + nighttime
                    "text_weekday_nighttime_mean_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_nighttime_mean_interaction_count": [
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                    "text_weekday_nighttime_min_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_nighttime_min_interaction_count": [
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                    "text_weekday_nighttime_max_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_nighttime_max_interaction_count": [
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                    "text_weekday_nighttime_std_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_nighttime_std_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_nighttime_median_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_nighttime_median_interaction_count": [
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                    "text_weekday_nighttime_skewness_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_nighttime_skewness_interaction_count": [
                        0.0,
                        0.0,
                        np.nan,
                        np.nan,
                    ],
                    "text_weekday_nighttime_kurtosis_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_nighttime_kurtosis_interaction_count": [
                        0.0,
                        0.0,
                        np.nan,
                        np.nan,
                    ],
                    # Only weekend dimension (weekend_allday)
                    "text_weekend_mean_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_mean_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "text_weekend_min_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_min_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "text_weekend_max_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_max_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "text_weekend_std_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_std_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "text_weekend_median_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_median_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "text_weekend_skewness_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_skewness_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "text_weekend_kurtosis_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_kurtosis_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    # Weekend + daytime
                    "text_weekend_daytime_mean_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_daytime_mean_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "text_weekend_daytime_min_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_daytime_min_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "text_weekend_daytime_max_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_daytime_max_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "text_weekend_daytime_std_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_daytime_std_interaction_count": [0.0, 0.0, 0.0, 0.0],
                    "text_weekend_daytime_median_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_daytime_median_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_daytime_skewness_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_daytime_skewness_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_daytime_kurtosis_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_daytime_kurtosis_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    # Weekend + nighttime
                    "text_weekend_nighttime_mean_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_nighttime_mean_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_nighttime_min_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_nighttime_min_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_nighttime_max_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_nighttime_max_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_nighttime_std_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_nighttime_std_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_nighttime_median_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_nighttime_median_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_nighttime_skewness_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_nighttime_skewness_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_nighttime_kurtosis_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_nighttime_kurtosis_interaction_count": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                }
            case "get_inter_event_time_stats":
                expected_results = {
                    "weekday_nighttime_text_mean_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_nighttime_call_mean_inter_event_time": [
                        0.0,
                        0.0,
                        np.nan,
                        np.nan,
                    ],
                    "weekend_nighttime_text_mean_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_nighttime_call_mean_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_text_mean_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_call_mean_inter_event_time": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_text_mean_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekend_daytime_call_mean_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekday_nighttime_text_min_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekday_nighttime_call_min_inter_event_time": [
                        0.0,
                        0.0,
                        np.nan,
                        np.nan,
                    ],
                    "weekend_nighttime_text_min_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekend_nighttime_call_min_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_text_min_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_call_min_inter_event_time": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_text_min_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekend_daytime_call_min_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekday_nighttime_text_max_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekday_nighttime_call_max_inter_event_time": [
                        0.0,
                        0.0,
                        np.nan,
                        np.nan,
                    ],
                    "weekend_nighttime_text_max_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekend_nighttime_call_max_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_text_max_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_call_max_inter_event_time": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_text_max_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekend_daytime_call_max_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekday_nighttime_text_std_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekday_nighttime_call_std_inter_event_time": [
                        0.0,
                        0.0,
                        np.nan,
                        np.nan,
                    ],
                    "weekend_nighttime_text_std_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekend_nighttime_call_std_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_text_std_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_call_std_inter_event_time": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_text_std_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekend_daytime_call_std_inter_event_time": [0.0, 0.0, 0.0, 0.0],
                    "weekday_nighttime_text_median_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_nighttime_call_median_inter_event_time": [
                        0.0,
                        0.0,
                        np.nan,
                        np.nan,
                    ],
                    "weekend_nighttime_text_median_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_nighttime_call_median_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_text_median_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_call_median_inter_event_time": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_text_median_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_call_median_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_nighttime_text_skewness_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_nighttime_call_skewness_inter_event_time": [
                        0.0,
                        0.0,
                        np.nan,
                        np.nan,
                    ],
                    "weekend_nighttime_text_skewness_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_nighttime_call_skewness_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_text_skewness_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_call_skewness_inter_event_time": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_text_skewness_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_call_skewness_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_nighttime_text_kurtosis_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_nighttime_call_kurtosis_inter_event_time": [
                        0.0,
                        0.0,
                        np.nan,
                        np.nan,
                    ],
                    "weekend_nighttime_text_kurtosis_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_nighttime_call_kurtosis_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_text_kurtosis_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_call_kurtosis_inter_event_time": [
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_text_kurtosis_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_call_kurtosis_inter_event_time": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                }

            case "get_pareto_principle_interaction_stats":
                expected_results = {
                    "weekday_nighttime_text_pareto_principle_interaction_fraction": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_nighttime_call_pareto_principle_interaction_fraction": [
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                    "weekend_nighttime_text_pareto_principle_interaction_fraction": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_nighttime_call_pareto_principle_interaction_fraction": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_text_pareto_principle_interaction_fraction": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_call_pareto_principle_interaction_fraction": [
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_text_pareto_principle_interaction_fraction": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_call_pareto_principle_interaction_fraction": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                }

            case "get_pareto_principle_call_duration_stats":
                expected_results = {
                    "weekday_nighttime_pareto_call_duration_fraction": [0, 0, 1, 1],
                    "weekend_nighttime_pareto_call_duration_fraction": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_pareto_call_duration_fraction": [
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_pareto_call_duration_fraction": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                }

            case "get_number_of_interactions_per_user":
                expected_results = {
                    "text_weekday_nighttime_incoming_num_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_daytime_incoming_num_interactions": [
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_nighttime_incoming_num_interactions": [
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                    "call_weekday_daytime_incoming_num_interactions": [
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_nighttime_incoming_num_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_daytime_incoming_num_interactions": [
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                    ],
                    "call_weekend_nighttime_incoming_num_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_daytime_incoming_num_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_nighttime_outgoing_num_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_daytime_outgoing_num_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_nighttime_outgoing_num_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekday_daytime_outgoing_num_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_nighttime_outgoing_num_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekend_daytime_outgoing_num_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_nighttime_outgoing_num_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "call_weekend_daytime_outgoing_num_interactions": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "text_weekday_nighttime_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_weekday_daytime_num_interactions": [0.0, 1.0, 0.0, 0.0],
                    "call_weekday_nighttime_num_interactions": [0.0, 0.0, 1.0, 1.0],
                    "call_weekday_daytime_num_interactions": [1.0, 1.0, 0.0, 0.0],
                    "text_weekend_nighttime_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_weekend_daytime_num_interactions": [0.0, 0.0, 1.0, 0.0],
                    "call_weekend_nighttime_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_daytime_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_nighttime_incoming_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "call_nighttime_incoming_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_daytime_incoming_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "call_daytime_incoming_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_nighttime_outgoing_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "call_nighttime_outgoing_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_daytime_outgoing_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "call_daytime_outgoing_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_nighttime_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "call_nighttime_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_daytime_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "call_daytime_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_weekday_incoming_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_incoming_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_weekend_incoming_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_incoming_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_weekday_outgoing_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_outgoing_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_weekend_outgoing_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_outgoing_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_weekday_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "call_weekday_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_weekend_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "call_weekend_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_incoming_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "call_incoming_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_outgoing_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "call_outgoing_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "text_num_interactions": [0.0, 0.0, 0.0, 0.0],
                    "call_num_interactions": [0.0, 0.0, 0.0, 0.0],
                }

            case "get_number_of_antennas":
                expected_results = {
                    "weekday_nighttime_num_unique_antennas": [0.0, 0.0, 0.0, 1.0],
                    "weekend_nighttime_num_unique_antennas": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_num_unique_antennas": [2.0, 1.0, 0.0, 0.0],
                    "weekend_daytime_num_unique_antennas": [0.0, 0.0, 1.0, 0.0],
                    "weekday_num_unique_antennas": [2.0, 1.0, 0.0, 1.0],
                    "weekend_num_unique_antennas": [0.0, 0.0, 1.0, 0.0],
                    "nighttime_num_unique_antennas": [0.0, 0.0, 0.0, 1.0],
                    "daytime_num_unique_antennas": [2.0, 1.0, 1.0, 0.0],
                    "num_unique_antennas": [2.0, 1.0, 1.0, 1.0],
                }

            case "get_entropy_of_antennas_per_caller":
                expected_results = {
                    "weekday_nighttime_entropy_of_antennas": [0.0, 0.0, 0.0, 0.0],
                    "weekend_nighttime_entropy_of_antennas": [0.0, 0.0, 0.0, 0.0],
                    "weekday_daytime_entropy_of_antennas": [
                        0.6931471805599453,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_entropy_of_antennas": [0.0, 0.0, 0.0, 0.0],
                    "weekday_entropy_of_antennas": [
                        0.6931471805599453,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_entropy_of_antennas": [0.0, 0.0, 0.0, 0.0],
                    "nighttime_entropy_of_antennas": [0.0, 0.0, 0.0, 0.0],
                    "daytime_entropy_of_antennas": [
                        0.6931471805599453,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "entropy_of_antennas": [
                        0.6931471805599453,
                        0.0,
                        0.0,
                        0.0,
                    ],
                }

            case "get_pareto_principle_antennas":
                expected_results = {
                    "weekday_nighttime_num_pareto_principle_antennas": [
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                    "weekend_nighttime_num_pareto_principle_antennas": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_daytime_num_pareto_principle_antennas": [
                        2.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    "weekend_daytime_num_pareto_principle_antennas": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "weekday_num_pareto_principle_antennas": [
                        2.0,
                        1.0,
                        1.0,
                        1.0,
                    ],
                    "weekend_num_pareto_principle_antennas": [0.0, 0.0, 0.0, 0.0],
                    "nighttime_num_pareto_principle_antennas": [0.0, 0.0, 1.0, 1.0],
                    "daytime_num_pareto_principle_antennas": [
                        2.0,
                        1.0,
                        0.0,
                        0.0,
                    ],
                    "num_pareto_principle_antennas": [2.0, 1.0, 1.0, 1.0],
                }
            case "get_average_num_of_interactions_from_home_antennas":
                expected_results = {
                    "weekday_nighttime_mean_home_antenna_interaction": [0.0, 0.0],
                    "weekend_nighttime_mean_home_antenna_interaction": [0.0, 0.0],
                    "weekday_daytime_mean_home_antenna_interaction": [0.0, 0.0],
                    "weekend_daytime_mean_home_antenna_interaction": [0.0, 0.0],
                    "weekday_mean_home_antenna_interaction": [0.0, 0.0],
                    "weekend_mean_home_antenna_interaction": [0.0, 0.0],
                    "nighttime_mean_home_antenna_interaction": [0.0, 0.0],
                    "daytime_mean_home_antenna_interaction": [0.0, 0.0],
                    "mean_home_antenna_interaction": [0.0, 0.0],
                }

            case "get_international_interaction_statistics":
                expected_results = {
                    "text_num_interactions": [1.0, 1.0],
                    "call_num_interactions": [1.0, 1.0],
                    "text_num_unique_recipients": [1.0, 1.0],
                    "call_num_unique_recipients": [1.0, 1.0],
                    "call_total_call_duration": [100.0, 100.0],
                    "text_num_unique_days": [1.0, 1.0],
                    "call_num_unique_days": [1.0, 1.0],
                    "num_unique_days": [1.0, 1.0],
                    "num_interactions": [2.0, 2.0],
                    "num_unique_recipients": [2.0, 2.0],
                }

            case _:
                raise ValueError(f"Function {function} does not exist.")

        return pd.DataFrame(expected_results).reset_index(drop=True)

    @pytest.mark.parametrize(
        "function_to_test",
        [
            "get_active_days",
            "get_number_of_contacts_per_caller",
            "get_call_duration_stats",
            "get_percentage_of_nocturnal_interactions",
            "get_percentage_of_initiated_conversations",
            "get_percentage_of_initiated_calls",
            "get_text_response_time_delay_stats",
            "get_text_response_rate",
            "get_entropy_of_interactions_per_caller",
            "get_outgoing_interaction_fraction_stats",
            "get_interaction_stats_per_caller",
            "get_inter_event_time_stats",
            "get_pareto_principle_interaction_stats",
            "get_pareto_principle_call_duration_stats",
            "get_number_of_interactions_per_user",
            "get_number_of_antennas",
            "get_entropy_of_antennas_per_caller",
            "get_pareto_principle_antennas",
            "get_average_num_of_interactions_from_home_antennas",
            "get_international_interaction_statistics",
        ],
    )
    def test_featurize_function(
        self, spark_cdr_with_conversations, spark, function_to_test
    ):
        expected_results = self._get_expected_results(function_to_test)
        func = self.FUNCTION_MAP[function_to_test]
        spark_function_output = func(spark_cdr_with_conversations)
        pd_function_output = spark_function_output.toPandas()

        assert set(
            [
                "caller_id",
                *expected_results.keys(),
            ]
        ) == set(pd_function_output.columns)
        assert (
            deepdiff.DeepDiff(
                pd_function_output.reset_index(drop=True).drop(columns=["caller_id"]),
                expected_results,
                ignore_order=True,
            )
            == {}
        )

        pd_cdr_with_conversations = spark_cdr_with_conversations.toPandas()
        for col in [
            k
            for k, field in CallDataRecordTagged.model_fields.items()
            if field.is_required()
        ]:
            spark_cdr_no_col = spark.createDataFrame(
                pd_cdr_with_conversations.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match=f"The following required columns are missing from the dataframe: {set([col])}",
            ):
                func(spark_cdr_no_col)

    def test_get_radius_of_gyration(self, spark_cdr_with_conversations, spark):
        pd_antenna_data = pd.DataFrame(ANTENNA_DATA)
        pd_antenna_data.rename(
            columns={"antenna_id": "caller_antenna_id"}, inplace=True
        )
        spark_antenna_data = spark.createDataFrame(pd_antenna_data)

        spark_radius_of_gyration = get_radius_of_gyration(
            spark_cdr_with_conversations, spark_antenna_data
        )
        pd_radius_of_gyration = spark_radius_of_gyration.toPandas()
        expected_results = {
            "weekday_nighttime_radius_of_gyration": [0.0, 0.0, 0.0, 0.0],
            "weekend_nighttime_radius_of_gyration": [0.0, 0.0, 0.0, 0.0],
            "weekday_daytime_radius_of_gyration": [327.0168802664289, 0.0, 0.0, 0.0],
            "weekend_daytime_radius_of_gyration": [0.0, 0.0, 0.0, 0.0],
            "radius_of_gyration": [327.0168802664289, 0.0, 0.0, 0.0],
            "weekend_radius_of_gyration": [0.0, 0.0, 0.0, 0.0],
            "weekday_radius_of_gyration": [327.0168802664289, 0.0, 0.0, 0.0],
            "nighttime_radius_of_gyration": [0.0, 0.0, 0.0, 0.0],
            "daytime_radius_of_gyration": [327.0168802664289, 0.0, 0.0, 0.0],
        }
        expected_results = pd.DataFrame(expected_results).reset_index(drop=True)

        assert set(
            [
                "caller_id",
                *expected_results.keys(),
            ]
        ) == set(pd_radius_of_gyration.columns)
        assert (
            deepdiff.DeepDiff(
                pd_radius_of_gyration.reset_index(drop=True).drop(
                    columns=["caller_id"]
                ),
                expected_results,
                ignore_order=True,
            )
            == {}
        )

        pd_cdr_with_conversations = spark_cdr_with_conversations.toPandas()
        for col in [
            k
            for k, field in CallDataRecordTagged.model_fields.items()
            if field.is_required()
        ]:
            spark_cdr_no_col = spark.createDataFrame(
                pd_cdr_with_conversations.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match=f"The following required columns are missing from the dataframe: {set([col])}",
            ):
                get_radius_of_gyration(spark_cdr_no_col, spark_antenna_data)

        for col in [
            k
            for k, field in AntennaDataGeometry.model_fields.items()
            if field.is_required()
        ]:
            spark_antenna_no_col = spark.createDataFrame(
                pd_antenna_data.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match=f"The following required columns are missing from the dataframe: {set([col])}",
            ):
                get_radius_of_gyration(
                    spark_cdr_with_conversations, spark_antenna_no_col
                )

    def test_get_caller_counts_per_region(self, spark_cdr_with_conversations, spark):
        pd_antenna_data = pd.DataFrame(ANTENNA_DATA)
        pd_antenna_data["region"] = ["RegionA", "RegionB", "RegionC"]
        pd_antenna_data.rename(
            columns={"antenna_id": "caller_antenna_id"}, inplace=True
        )

        spark_antenna_data = spark.createDataFrame(pd_antenna_data)

        spark_caller_counts_per_region = get_caller_counts_per_region(
            spark_cdr_with_conversations, spark_antenna_data
        )
        pd_caller_counts_per_region = spark_caller_counts_per_region.toPandas()

        expected_results = {
            "RegionA_first(num_unique_interactions)": [1.0, 1.0, np.nan, 1.0],
            "RegionA_first(num_unique_antennas)": [1.0, 1.0, np.nan, 1.0],
            "RegionB_first(num_unique_interactions)": [1.0, 1.0, np.nan, 1.0],
            "RegionB_first(num_unique_antennas)": [1.0, 1.0, np.nan, 1.0],
            "RegionC_first(num_unique_interactions)": [np.nan, np.nan, 3.0, np.nan],
            "RegionC_first(num_unique_antennas)": [np.nan, np.nan, 1.0, np.nan],
        }
        expected_results = pd.DataFrame(expected_results).reset_index(drop=True)

        assert set(
            [
                "caller_id",
                *expected_results.keys(),
            ]
        ) == set(pd_caller_counts_per_region.columns)
        assert (
            deepdiff.DeepDiff(
                pd_caller_counts_per_region.reset_index(drop=True).drop(
                    columns=["caller_id"]
                ),
                expected_results,
                ignore_order=True,
            )
            == {}
        )

        pd_cdr_with_conversations = spark_cdr_with_conversations.toPandas()
        for col in [
            k
            for k, field in CallDataRecordTagged.model_fields.items()
            if field.is_required()
        ]:
            spark_cdr_no_col = spark.createDataFrame(
                pd_cdr_with_conversations.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match=f"The following required columns are missing from the dataframe: {set([col])}",
            ):
                get_caller_counts_per_region(spark_cdr_no_col, spark_antenna_data)

        for col in [
            k
            for k, field in AntennaDataGeometryWithRegion.model_fields.items()
            if field.is_required()
        ]:
            spark_antenna_no_col = spark.createDataFrame(
                pd_antenna_data.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match=f"The following required columns are missing from the dataframe: {set([col])}",
            ):
                get_caller_counts_per_region(
                    spark_cdr_with_conversations, spark_antenna_no_col
                )


class TestFeaturizerCoreMobileData:

    def test_get_mobile_data_stats(self, spark):
        pd_mobile_data = pd.DataFrame(MOBILE_DATA_USAGE_DATA)
        pd_mobile_data.loc[:, "day"] = pd_mobile_data["timestamp"].dt.date

        spark_mobile_data = spark.createDataFrame(pd_mobile_data)

        spark_mobile_data_stats = get_mobile_data_stats(spark_mobile_data)
        pd_mobile_data_stats = spark_mobile_data_stats.toPandas()

        assert pd_mobile_data_stats.shape == (3, 7)
        assert set(
            [
                "caller_id",
                "total_data_volume",
                "mean_daily_data_volume",
                "min_daily_data_volume",
                "max_daily_data_volume",
                "std_daily_data_volume",
                "num_unique_days_with_data_usage",
            ]
        ) == set(pd_mobile_data_stats.columns)

        for col in [
            k
            for k, field in MobileDataUsageDataWithDay.model_fields.items()
            if field.is_required()
        ]:
            spark_mobile_data_no_col = spark.createDataFrame(
                pd_mobile_data.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match=f"The following required columns are missing from the dataframe: {set([col])}",
            ):
                get_mobile_data_stats(spark_mobile_data_no_col)


class TestFeaturizerCoreMobileMoney:

    FUNCTION_MAP = {
        "get_mobile_money_amount_stats": get_mobile_money_amount_stats,
        "get_mobile_money_transaction_stats": get_mobile_money_transaction_stats,
        "get_mobile_money_balance_stats": get_mobile_money_balance_stats,
    }

    @pytest.fixture
    def spark_mobile_money_with_direction(self, spark):
        pd_mobile_money_data = pd.DataFrame(MOBILE_MONEY_TRANSACTION_DATA)
        pd_mobile_money_data.loc[:, "day"] = pd_mobile_money_data["timestamp"].dt.date

        spark_mobile_money_data = spark.createDataFrame(pd_mobile_money_data)
        spark_mobile_money_direction = identify_mobile_money_transaction_direction(
            spark_mobile_money_data
        )
        return spark_mobile_money_direction

    def _get_expected_results(self, function):
        match function:
            case "get_mobile_money_amount_stats":
                expected_results = {
                    "billpay_first(mean_amount)": [
                        1500.0,
                        np.nan,
                        1500.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "billpay_first(min_amount)": [
                        1500.0,
                        np.nan,
                        1500.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "billpay_first(max_amount)": [
                        1500.0,
                        np.nan,
                        1500.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "billpay_first(std_amount)": [
                        0.0,
                        np.nan,
                        0.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "cashin_first(mean_amount)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        1000.0,
                        1000.0,
                        np.nan,
                    ],
                    "cashin_first(min_amount)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        1000.0,
                        1000.0,
                        np.nan,
                    ],
                    "cashin_first(max_amount)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        1000.0,
                        1000.0,
                        np.nan,
                    ],
                    "cashin_first(std_amount)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                        np.nan,
                    ],
                    "cashout_first(mean_amount)": [
                        1500.0,
                        np.nan,
                        1500.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "cashout_first(min_amount)": [
                        1500.0,
                        np.nan,
                        1500.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "cashout_first(max_amount)": [
                        1500.0,
                        np.nan,
                        1500.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "cashout_first(std_amount)": [
                        0.0,
                        np.nan,
                        0.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "other_first(mean_amount)": [
                        np.nan,
                        2000.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        2000.0,
                    ],
                    "other_first(min_amount)": [
                        np.nan,
                        2000.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        2000.0,
                    ],
                    "other_first(max_amount)": [
                        np.nan,
                        2000.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        2000.0,
                    ],
                    "other_first(std_amount)": [
                        np.nan,
                        0.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        0.0,
                    ],
                    "p2p_first(mean_amount)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        1000.0,
                        1000.0,
                        np.nan,
                    ],
                    "p2p_first(min_amount)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        1000.0,
                        1000.0,
                        np.nan,
                    ],
                    "p2p_first(max_amount)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        1000.0,
                        1000.0,
                        np.nan,
                    ],
                    "p2p_first(std_amount)": [np.nan, np.nan, np.nan, 0.0, 0.0, np.nan],
                    "mean_amount": [1500.0, 2000.0, 1500.0, 1000.0, 1000.0, 2000.0],
                    "min_amount": [1500.0, 2000.0, 1500.0, 1000.0, 1000.0, 2000.0],
                    "max_amount": [1500.0, 2000.0, 1500.0, 1000.0, 1000.0, 2000.0],
                    "std_amount": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }

            case "get_mobile_money_transaction_stats":
                expected_results = {
                    "billpay_first(num_transactions)": [
                        1.0,
                        np.nan,
                        1.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "billpay_first(num_unique_correspondents)": [
                        1.0,
                        np.nan,
                        1.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "cashin_first(num_transactions)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        1.0,
                        1.0,
                        np.nan,
                    ],
                    "cashin_first(num_unique_correspondents)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        1.0,
                        1.0,
                        np.nan,
                    ],
                    "cashout_first(num_transactions)": [
                        1.0,
                        np.nan,
                        1.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "cashout_first(num_unique_correspondents)": [
                        1.0,
                        np.nan,
                        1.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "other_first(num_transactions)": [
                        np.nan,
                        2.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        2.0,
                    ],
                    "other_first(num_unique_correspondents)": [
                        np.nan,
                        1.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        1.0,
                    ],
                    "p2p_first(num_transactions)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        1.0,
                        1.0,
                        np.nan,
                    ],
                    "p2p_first(num_unique_correspondents)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        1.0,
                        1.0,
                        np.nan,
                    ],
                }

            case "get_mobile_money_balance_stats":
                expected_results = {
                    "billpay_first(mean_balance_after)": [
                        4500.0,
                        np.nan,
                        4000.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "billpay_first(min_balance_after)": [
                        4500.0,
                        np.nan,
                        4000.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "billpay_first(max_balance_after)": [
                        4500.0,
                        np.nan,
                        4000.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "billpay_first(std_balance_after)": [
                        0.0,
                        np.nan,
                        0.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "billpay_first(mean_balance_before)": [
                        6000.0,
                        np.nan,
                        2500.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "billpay_first(min_balance_before)": [
                        6000.0,
                        np.nan,
                        2500.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "billpay_first(max_balance_before)": [
                        6000.0,
                        np.nan,
                        2500.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "billpay_first(std_balance_before)": [
                        0.0,
                        np.nan,
                        0.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "cashin_first(mean_balance_after)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        4000.0,
                        3000.0,
                        np.nan,
                    ],
                    "cashin_first(min_balance_after)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        4000.0,
                        3000.0,
                        np.nan,
                    ],
                    "cashin_first(max_balance_after)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        4000.0,
                        3000.0,
                        np.nan,
                    ],
                    "cashin_first(std_balance_after)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                        np.nan,
                    ],
                    "cashin_first(mean_balance_before)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        5000.0,
                        2000.0,
                        np.nan,
                    ],
                    "cashin_first(min_balance_before)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        5000.0,
                        2000.0,
                        np.nan,
                    ],
                    "cashin_first(max_balance_before)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        5000.0,
                        2000.0,
                        np.nan,
                    ],
                    "cashin_first(std_balance_before)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                        np.nan,
                    ],
                    "cashout_first(mean_balance_after)": [
                        4500.0,
                        np.nan,
                        4000.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "cashout_first(min_balance_after)": [
                        4500.0,
                        np.nan,
                        4000.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "cashout_first(max_balance_after)": [
                        4500.0,
                        np.nan,
                        4000.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "cashout_first(std_balance_after)": [
                        0.0,
                        np.nan,
                        0.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "cashout_first(mean_balance_before)": [
                        6000.0,
                        np.nan,
                        2500.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "cashout_first(min_balance_before)": [
                        6000.0,
                        np.nan,
                        2500.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "cashout_first(max_balance_before)": [
                        6000.0,
                        np.nan,
                        2500.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "cashout_first(std_balance_before)": [
                        0.0,
                        np.nan,
                        0.0,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "other_first(mean_balance_after)": [
                        np.nan,
                        5000.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        5000.0,
                    ],
                    "other_first(min_balance_after)": [
                        np.nan,
                        5000.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        5000.0,
                    ],
                    "other_first(max_balance_after)": [
                        np.nan,
                        5000.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        5000.0,
                    ],
                    "other_first(std_balance_after)": [
                        np.nan,
                        0.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        0.0,
                    ],
                    "other_first(mean_balance_before)": [
                        np.nan,
                        3000.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        7000.0,
                    ],
                    "other_first(min_balance_before)": [
                        np.nan,
                        3000.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        7000.0,
                    ],
                    "other_first(max_balance_before)": [
                        np.nan,
                        3000.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        7000.0,
                    ],
                    "other_first(std_balance_before)": [
                        np.nan,
                        0.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        0.0,
                    ],
                    "p2p_first(mean_balance_after)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        4000.0,
                        3000.0,
                        np.nan,
                    ],
                    "p2p_first(min_balance_after)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        4000.0,
                        3000.0,
                        np.nan,
                    ],
                    "p2p_first(max_balance_after)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        4000.0,
                        3000.0,
                        np.nan,
                    ],
                    "p2p_first(std_balance_after)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                        np.nan,
                    ],
                    "p2p_first(mean_balance_before)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        5000.0,
                        2000.0,
                        np.nan,
                    ],
                    "p2p_first(min_balance_before)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        5000.0,
                        2000.0,
                        np.nan,
                    ],
                    "p2p_first(max_balance_before)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        5000.0,
                        2000.0,
                        np.nan,
                    ],
                    "p2p_first(std_balance_before)": [
                        np.nan,
                        np.nan,
                        np.nan,
                        0.0,
                        0.0,
                        np.nan,
                    ],
                    "mean_balance_after": [
                        4500.0,
                        5000.0,
                        4000.0,
                        4000.0,
                        3000.0,
                        5000.0,
                    ],
                    "min_balance_after": [
                        4500.0,
                        5000.0,
                        4000.0,
                        4000.0,
                        3000.0,
                        5000.0,
                    ],
                    "max_balance_after": [
                        4500.0,
                        5000.0,
                        4000.0,
                        4000.0,
                        3000.0,
                        5000.0,
                    ],
                    "std_balance_after": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "mean_balance_before": [
                        6000.0,
                        3000.0,
                        2500.0,
                        5000.0,
                        2000.0,
                        7000.0,
                    ],
                    "min_balance_before": [
                        6000.0,
                        3000.0,
                        2500.0,
                        5000.0,
                        2000.0,
                        7000.0,
                    ],
                    "max_balance_before": [
                        6000.0,
                        3000.0,
                        2500.0,
                        5000.0,
                        2000.0,
                        7000.0,
                    ],
                    "std_balance_before": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }

            case _:
                raise ValueError(f"Function {function} does not exist.")

        return pd.DataFrame(expected_results).reset_index(drop=True)

    @pytest.mark.parametrize(
        "function_to_test",
        [
            "get_mobile_money_amount_stats",
            "get_mobile_money_transaction_stats",
            "get_mobile_money_balance_stats",
        ],
    )
    def test_featurize_function(
        self, spark_mobile_money_with_direction, spark, function_to_test
    ):
        expected_results = self._get_expected_results(function_to_test)
        func = self.FUNCTION_MAP[function_to_test]
        spark_function_output = func(spark_mobile_money_with_direction)
        pd_function_output = spark_function_output.toPandas()

        assert set(
            [
                "primary_id",
                *expected_results.keys(),
            ]
        ) == set(pd_function_output.columns)
        assert (
            deepdiff.DeepDiff(
                pd_function_output.reset_index(drop=True).drop(columns=["primary_id"]),
                expected_results,
                ignore_order=True,
            )
            == {}
        )

        pd_mobile_money_with_direction = spark_mobile_money_with_direction.toPandas()
        for col in [
            k
            for k, field in MobileMoneyDataWithDirection.model_fields.items()
            if field.is_required()
        ]:
            spark_mobile_money_no_col = spark.createDataFrame(
                pd_mobile_money_with_direction.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match=f"The following required columns are missing from the dataframe: {set([col])}",
            ):
                func(spark_mobile_money_no_col)


class TestFeaturizerCoreRechargeData:

    def test_get_recharge_amount_stats(self, spark):
        pd_recharge_data = pd.DataFrame(RECHARGE_DATA)
        pd_recharge_data.loc[:, "day"] = pd_recharge_data["timestamp"].dt.date

        spark_recharge_data = spark.createDataFrame(pd_recharge_data)

        spark_recharge_data_stats = get_recharge_amount_stats(spark_recharge_data)
        pd_recharge_data_stats = spark_recharge_data_stats.toPandas()

        expected_results = {
            "total_recharge_amount": [300.0, 200.0, 400.0],
            "num_recharges": [2.0, 2.0, 2.0],
            "num_unique_recharge_days": [2.0, 2.0, 2.0],
            "mean_amount": [150.0, 100.0, 200.0],
            "min_amount": [150.0, 100.0, 200.0],
            "max_amount": [150.0, 100.0, 200.0],
        }
        expected_results = pd.DataFrame(expected_results).reset_index(drop=True)
        assert set(["caller_id", *expected_results.columns]) == set(
            pd_recharge_data_stats.columns
        )
        assert (
            deepdiff.DeepDiff(
                pd_recharge_data_stats.reset_index(drop=True).drop(
                    columns=["caller_id"]
                ),
                expected_results,
                ignore_order=True,
            )
            == {}
        )

        for col in [
            k
            for k, field in RechargeDataWithDay.model_fields.items()
            if field.is_required()
        ]:
            spark_recharge_data_no_col = spark.createDataFrame(
                pd_recharge_data.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match=f"The following required columns are missing from the dataframe: {set([col])}",
            ):
                get_recharge_amount_stats(spark_recharge_data_no_col)
