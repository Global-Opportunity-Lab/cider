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
import pandas as pd
import pytest
from cider.featurizer.dependencies import (
    filter_to_datetime,
    get_spammers_from_cdr_data,
    get_outlier_days_from_cdr_data,
    get_static_diagnostic_statistics,
    get_timeseries_diagnostic_statistics,
)
from cider.featurizer.inference import (
    identify_daytime,
    identify_weekend,
    swap_caller_and_recipient,
    identify_and_tag_conversations,
    identify_active_days,
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
    get_radius_of_gyration,
    get_pareto_principle_antennas,
    get_average_num_of_interactions_from_home_antennas,
)


class TestFeaturizerDependencies:

    @pytest.mark.parametrize(
        "dataset",
        [
            CDR_DATA,
            MOBILE_DATA_USAGE_DATA,
            MOBILE_MONEY_TRANSACTION_DATA,
            RECHARGE_DATA,
        ],
    )
    def test_filter_to_datetime(self, dataset):
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
        assert len(filtered_data) == 2

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
        assert stats.num_days == df["timestamp"].dt.date.nunique()
        if "recipient_id" in df.columns:
            assert stats.num_unique_recipients == df["recipient_id"].nunique()
        else:
            assert stats.num_unique_recipients == 0

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


class TestFeaturizerInference:

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
        spark_cdr_data = spark.createDataFrame(pd.DataFrame(CDR_DATA))
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
            "caller_id",
            "recipient_id",
            "caller_antenna_id",
            "recipient_antenna_id",
        ]:
            spark_cdr_no_col = spark.createDataFrame(pd_cdr_data.drop(columns=[col]))
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'recipient_id', 'caller_antenna_id', and 'recipient_antenna_id' columns",
            ):
                swap_caller_and_recipient(spark_cdr_no_col)

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
            "transaction_scope": ["domestic"] * 6,
            "transaction_type": ["text", "text", "call", "text", "text", "text"],
        }
        pd_cdr_data = pd.concat(
            [pd.DataFrame(CDR_DATA), pd.DataFrame(conversations)], ignore_index=True
        )
        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_tagged = identify_and_tag_conversations(spark_cdr_data, max_wait=3600)
        pd_cdr_tagged = spark_cdr_tagged.toPandas()

        assert "conversation" in pd_cdr_tagged.columns
        convo_times = pd_cdr_tagged["conversation"].dropna().unique()
        assert len(convo_times) == 5

        for col in ["caller_id", "recipient_id", "timestamp", "transaction_type"]:
            spark_cdr_no_col = spark.createDataFrame(pd_cdr_data.drop(columns=[col]))
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'recipient_id', 'timestamp', and 'transaction_type' columns",
            ):
                identify_and_tag_conversations(spark_cdr_no_col)

    def test_identify_active_days(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)
        pd_cdr_data.loc[:, "day"] = pd_cdr_data["timestamp"].dt.date

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)

        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_cdr_with_conversations = identify_and_tag_conversations(
            spark_cdr_with_weekend
        )
        spark_cdr_active_days = identify_active_days(spark_cdr_with_conversations)

        pd_cdr_active_days = spark_cdr_active_days.toPandas()

        assert set(
            [
                "active_days_all",
                "active_days_weekday",
                "active_days_weekend",
                "active_days_day",
                "active_days_night",
                "active_days_weekday_day",
                "active_days_weekday_night",
                "active_days_weekend_day",
                "active_days_weekend_night",
            ]
        ).issubset(set(pd_cdr_active_days.columns))
        assert pd_cdr_active_days.shape == (3, 10)
        assert pd_cdr_active_days.active_days_all.values.tolist() == [2, 2, 2]
        assert pd_cdr_active_days.active_days_weekday.values.tolist() == [2, 1, 2]
        assert pd_cdr_active_days.active_days_weekend.values.tolist() == [0, 1, 0]
        assert pd_cdr_active_days.active_days_day.values.tolist() == [2, 2, 1]
        assert pd_cdr_active_days.active_days_night.values.tolist() == [0, 0, 1]
        assert pd_cdr_active_days.active_days_weekday_day.values.tolist() == [2, 1, 1]
        assert pd_cdr_active_days.active_days_weekday_night.values.tolist() == [0, 0, 1]
        assert pd_cdr_active_days.active_days_weekend_day.values.tolist() == [0, 1, 0]
        assert pd_cdr_active_days.active_days_weekend_night.values.tolist() == [0, 0, 0]

        pd_cdr_with_conversations = spark_cdr_with_conversations.toPandas()
        for col in ["caller_id", "timestamp", "is_daytime", "is_weekend", "day"]:
            spark_cdr_no_col = spark.createDataFrame(
                pd_cdr_with_conversations.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'timestamp', 'day', 'is_weekend', and 'is_daytime' columns",
            ):
                identify_active_days(spark_cdr_no_col)

    def test_get_number_of_contacts_per_caller(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)

        spark_cdr_num_contacts = get_number_of_contacts_per_caller(
            spark_cdr_with_weekend
        )

        pd_cdr_num_contacts = spark_cdr_num_contacts.toPandas()

        assert pd_cdr_num_contacts.shape == (3, 9)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_text_num_unique_contacts",
                "weekday_daytime_text_num_unique_contacts",
                "weekday_nighttime_call_num_unique_contacts",
                "weekday_daytime_call_num_unique_contacts",
                "weekend_nighttime_text_num_unique_contacts",
                "weekend_daytime_text_num_unique_contacts",
                "weekend_nighttime_call_num_unique_contacts",
                "weekend_daytime_call_num_unique_contacts",
            ]
        ) == set(pd_cdr_num_contacts.columns)

        pd_cdr_with_weekend = spark_cdr_with_weekend.toPandas()
        for col in [
            "caller_id",
            "recipient_id",
            "is_weekend",
            "is_daytime",
            "transaction_type",
        ]:
            spark_cdr_data_missing = spark.createDataFrame(
                pd_cdr_with_weekend.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'recipient_id', 'is_weekend', 'is_daytime', and 'transaction_type' columns",
            ):
                get_number_of_contacts_per_caller(spark_cdr_data_missing)

    def test_get_call_duration_stats(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)

        spark_cdr_call_stats = get_call_duration_stats(spark_cdr_with_weekend)

        pd_cdr_call_stats = spark_cdr_call_stats.toPandas()

        assert pd_cdr_call_stats.shape == (3, 29)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_mean_duration",
                "weekend_nighttime_mean_duration",
                "weekday_daytime_mean_duration",
                "weekend_daytime_mean_duration",
                "weekday_nighttime_median_duration",
                "weekend_nighttime_median_duration",
                "weekday_daytime_median_duration",
                "weekend_daytime_median_duration",
                "weekday_nighttime_max_duration",
                "weekend_nighttime_max_duration",
                "weekday_daytime_max_duration",
                "weekend_daytime_max_duration",
                "weekday_nighttime_min_duration",
                "weekend_nighttime_min_duration",
                "weekday_daytime_min_duration",
                "weekend_daytime_min_duration",
                "weekday_nighttime_std_duration",
                "weekend_nighttime_std_duration",
                "weekday_daytime_std_duration",
                "weekend_daytime_std_duration",
                "weekday_nighttime_skewness_duration",
                "weekend_nighttime_skewness_duration",
                "weekday_daytime_skewness_duration",
                "weekend_daytime_skewness_duration",
                "weekday_nighttime_kurtosis_duration",
                "weekend_nighttime_kurtosis_duration",
                "weekday_daytime_kurtosis_duration",
                "weekend_daytime_kurtosis_duration",
            ]
        ) == set(pd_cdr_call_stats.columns)

        pd_cdr_with_weekend = spark_cdr_with_weekend.toPandas()
        for col in [
            "caller_id",
            "is_weekend",
            "is_daytime",
            "transaction_type",
            "duration",
        ]:
            spark_cdr_data_missing = spark.createDataFrame(
                pd_cdr_with_weekend.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'transaction_type', 'is_weekend', 'is_daytime', and 'duration' columns",
            ):
                get_call_duration_stats(spark_cdr_data_missing)

    def test_get_percentage_nocturnal_interactions(self, spark):
        cdr_data = {
            "caller_id": ["caller_1"] * 3 + ["caller_2"] * 3,
            "recipient_id": ["recipient_1"] * 6,
            "caller_antenna_id": ["antenna_1", "antenna_2"] * 3,
            "recipient_antenna_id": ["antenna_3", "antenna_4"] * 3,
            "timestamp": pd.to_datetime(
                [
                    "2023-01-01 10:00:00",
                    "2023-01-02 12:00:00",
                    "2023-01-02 14:00:00",
                    "2023-01-04 22:00:00",
                    "2023-01-05 18:00:00",
                    "2023-01-06 21:00:00",
                ]
            ),
            "duration": [300, 200, 400, 100, 250, 150],
            "transaction_type": ["text", "call"] * 3,
            "transaction_scope": ["domestic"] * 2
            + ["international"] * 2
            + ["other"] * 2,
        }
        pd_cdr_data = pd.DataFrame(cdr_data)

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)

        spark_cdr_nocturnal_calls = get_percentage_of_nocturnal_interactions(
            spark_cdr_with_weekend
        )

        pd_cdr_nocturnal_calls = spark_cdr_nocturnal_calls.toPandas()

        assert pd_cdr_nocturnal_calls.shape == (2, 5)
        assert pd_cdr_nocturnal_calls.filter(like="nocturnal").sum(
            1
        ).tolist() == pytest.approx([66.67, 0.0], rel=1e-2)
        assert set(
            [
                "caller_id",
                "weekday_text_percentage_nocturnal_interactions",
                "weekend_text_percentage_nocturnal_interactions",
                "weekday_call_percentage_nocturnal_interactions",
                "weekend_call_percentage_nocturnal_interactions",
            ]
        ) == set(pd_cdr_nocturnal_calls.columns)

        pd_cdr_with_weekend = spark_cdr_with_weekend.toPandas()
        for col in ["caller_id", "is_daytime", "is_weekend", "transaction_type"]:
            spark_cdr_data_missing = spark.createDataFrame(
                pd_cdr_with_weekend.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'is_daytime', 'is_weekend' and 'transaction_type' columns",
            ):
                get_percentage_of_nocturnal_interactions(spark_cdr_data_missing)

    def test_get_percentage_of_initiated_conversations(self, spark):
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
            "transaction_scope": ["domestic"] * 6,
            "transaction_type": ["text", "text", "call", "text", "text", "text"],
        }
        pd_cdr_data = pd.concat(
            [pd.DataFrame(CDR_DATA), pd.DataFrame(conversations)], ignore_index=True
        )
        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_cdr_swapped = swap_caller_and_recipient(spark_cdr_with_weekend)
        spark_cdr_tagged = identify_and_tag_conversations(
            spark_cdr_swapped, max_wait=3600
        )
        spark_cdr_percentage_initiated = get_percentage_of_initiated_conversations(
            spark_cdr_tagged
        )
        pd_cdr_percentage_initiated = spark_cdr_percentage_initiated.toPandas()

        assert pd_cdr_percentage_initiated.shape == (6, 5)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_percentage_initiated_conversations",
                "weekend_nighttime_percentage_initiated_conversations",
                "weekday_daytime_percentage_initiated_conversations",
                "weekend_daytime_percentage_initiated_conversations",
            ]
        ) == set(pd_cdr_percentage_initiated.columns)

        pd_cdr_tagged = spark_cdr_tagged.toPandas()
        for col in [
            "caller_id",
            "timestamp",
            "conversation",
            "is_weekend",
            "is_daytime",
            "direction_of_transaction",
        ]:
            spark_cdr_no_col = spark.createDataFrame(pd_cdr_tagged.drop(columns=[col]))
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'timestamp', 'conversation', 'is_weekend', 'is_daytime' and 'direction_of_transaction' columns",
            ):
                get_percentage_of_initiated_conversations(spark_cdr_no_col)

    def test_get_percentage_of_initiated_calls(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)
        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_cdr_swapped = swap_caller_and_recipient(spark_cdr_with_weekend)
        spark_cdr_percentage_initiated = get_percentage_of_initiated_calls(
            spark_cdr_swapped
        )
        pd_cdr_percentage_initiated = spark_cdr_percentage_initiated.toPandas()
        assert pd_cdr_percentage_initiated.shape == (4, 5)
        assert pd_cdr_percentage_initiated.filter(like="initiated").sum(1).tolist() == [
            1.0,
            1.0,
            0.0,
            1.0,
        ]
        assert set(
            [
                "caller_id",
                "weekday_nighttime_percentage_initiated_calls",
                "weekend_nighttime_percentage_initiated_calls",
                "weekday_daytime_percentage_initiated_calls",
                "weekend_daytime_percentage_initiated_calls",
            ]
        ) == set(pd_cdr_percentage_initiated.columns)

        pd_cdr_swapped = spark_cdr_swapped.toPandas()
        for col in [
            "caller_id",
            "is_weekend",
            "is_daytime",
            "direction_of_transaction",
            "transaction_type",
        ]:
            spark_cdr_no_col = spark.createDataFrame(pd_cdr_swapped.drop(columns=[col]))
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'is_weekend', 'is_daytime', 'direction_of_transaction' and 'transaction_type' columns",
            ):
                get_percentage_of_initiated_calls(spark_cdr_no_col)

    def test_get_text_response_time_delay_stats(self, spark):
        conversations = {
            "caller_id": ["user_1"] * 6,
            "recipient_id": ["user_2"] * 6,
            "caller_antenna_id": ["antenna_1"] * 6,
            "recipient_antenna_id": ["antenna_2"] * 6,
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
            "transaction_scope": ["domestic"] * 6,
            "transaction_type": ["text", "text", "call", "text", "text", "text"],
        }
        pd_cdr_data = pd.DataFrame(conversations)
        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_cdr_swapped = swap_caller_and_recipient(spark_cdr_with_weekend)
        spark_cdr_tagged = identify_and_tag_conversations(
            spark_cdr_swapped, max_wait=3600
        )
        spark_cdr_text_response_time_delay = get_text_response_time_delay_stats(
            spark_cdr_tagged
        )
        pd_cdr_text_response_time_delay = spark_cdr_text_response_time_delay.toPandas()
        assert pd_cdr_text_response_time_delay.shape == (2, 29)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_mean_response_time_delay",
                "weekend_nighttime_mean_response_time_delay",
                "weekday_daytime_mean_response_time_delay",
                "weekend_daytime_mean_response_time_delay",
                "weekday_nighttime_median_response_time_delay",
                "weekend_nighttime_median_response_time_delay",
                "weekday_daytime_median_response_time_delay",
                "weekend_daytime_median_response_time_delay",
                "weekday_nighttime_max_response_time_delay",
                "weekend_nighttime_max_response_time_delay",
                "weekday_daytime_max_response_time_delay",
                "weekend_daytime_max_response_time_delay",
                "weekday_nighttime_min_response_time_delay",
                "weekend_nighttime_min_response_time_delay",
                "weekday_daytime_min_response_time_delay",
                "weekend_daytime_min_response_time_delay",
                "weekday_nighttime_std_response_time_delay",
                "weekend_nighttime_std_response_time_delay",
                "weekday_daytime_std_response_time_delay",
                "weekend_daytime_std_response_time_delay",
                "weekday_nighttime_skewness_response_time_delay",
                "weekend_nighttime_skewness_response_time_delay",
                "weekday_daytime_skewness_response_time_delay",
                "weekend_daytime_skewness_response_time_delay",
                "weekday_nighttime_kurtosis_response_time_delay",
                "weekend_nighttime_kurtosis_response_time_delay",
                "weekday_daytime_kurtosis_response_time_delay",
                "weekend_daytime_kurtosis_response_time_delay",
            ]
        ) == set(pd_cdr_text_response_time_delay.columns)

        pd_cdr_tagged = spark_cdr_tagged.toPandas()
        for col in [
            "caller_id",
            "recipient_id",
            "timestamp",
            "transaction_type",
            "conversation",
            "is_weekend",
            "is_daytime",
            "direction_of_transaction",
        ]:
            spark_cdr_no_col = spark.createDataFrame(pd_cdr_tagged.drop(columns=[col]))
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'recipient_id', 'transaction_type', 'timestamp', 'is_weekend', 'is_daytime', 'conversation', and 'direction_of_transaction' columns",
            ):
                get_text_response_time_delay_stats(spark_cdr_no_col)

    def test_get_text_response_rate(self, spark):
        conversations = {
            "caller_id": ["user_1"] * 3 + ["user_2"] * 3,
            "recipient_id": ["user_2"] * 3 + ["user_1"] * 3,
            "caller_antenna_id": ["antenna_1"] * 6,
            "recipient_antenna_id": ["antenna_2"] * 6,
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
            "transaction_scope": ["domestic"] * 6,
            "transaction_type": ["text", "text", "call", "text", "text", "text"],
        }
        pd_cdr_data = pd.DataFrame(conversations)
        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_cdr_swapped = swap_caller_and_recipient(spark_cdr_with_weekend)
        spark_cdr_tagged = identify_and_tag_conversations(
            spark_cdr_swapped, max_wait=3600
        )
        spark_cdr_text_response_rate = get_text_response_rate(spark_cdr_tagged)
        pd_cdr_text_response_rate = spark_cdr_text_response_rate.toPandas()

        assert pd_cdr_text_response_rate.shape == (2, 5)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_text_response_rate",
                "weekend_nighttime_text_response_rate",
                "weekday_daytime_text_response_rate",
                "weekend_daytime_text_response_rate",
            ]
        ) == set(pd_cdr_text_response_rate.columns)

        pd_cdr_tagged = spark_cdr_tagged.toPandas()
        for col in [
            "caller_id",
            "recipient_id",
            "timestamp",
            "transaction_type",
            "conversation",
            "is_weekend",
            "is_daytime",
            "direction_of_transaction",
        ]:
            spark_cdr_no_col = spark.createDataFrame(pd_cdr_tagged.drop(columns=[col]))
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'recipient_id', 'transaction_type', 'timestamp', 'is_weekend', 'is_daytime', 'conversation', and 'direction_of_transaction' columns",
            ):
                get_text_response_rate(spark_cdr_no_col)

    def test_get_entropy_of_interactions_per_caller(self, spark):
        cdr_data = CDR_DATA.copy()
        cdr_data["caller_id"] = ["caller_1"] * 6
        cdr_data["recipient_id"] = [
            f"recipient_{i}" for i in range(len(cdr_data["caller_id"]))
        ]
        pd_cdr_data = pd.DataFrame(cdr_data)

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_entropy_of_interactions = get_entropy_of_interactions_per_caller(
            spark_cdr_with_weekend
        )
        pd_cdr_entropy_of_interactions = spark_entropy_of_interactions.toPandas()

        assert pd_cdr_entropy_of_interactions.shape == (1, 9)
        assert pd_cdr_entropy_of_interactions.filter(like="entropy").sum(1)[
            0
        ] == pytest.approx(0.0, rel=1e-3)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_text_entropy_of_interactions",
                "weekday_nighttime_call_entropy_of_interactions",
                "weekend_nighttime_text_entropy_of_interactions",
                "weekend_nighttime_call_entropy_of_interactions",
                "weekday_daytime_text_entropy_of_interactions",
                "weekday_daytime_call_entropy_of_interactions",
                "weekend_daytime_text_entropy_of_interactions",
                "weekend_daytime_call_entropy_of_interactions",
            ]
        ) == set(pd_cdr_entropy_of_interactions.columns)

        pd_cdr_with_weekend = spark_cdr_with_weekend.toPandas()
        for col in [
            "caller_id",
            "recipient_id",
            "is_weekend",
            "is_daytime",
            "transaction_type",
        ]:
            spark_cdr_no_col = spark.createDataFrame(
                pd_cdr_with_weekend.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'recipient_id', 'is_weekend', 'is_daytime', and 'transaction_type' columns",
            ):
                get_entropy_of_interactions_per_caller(spark_cdr_no_col)

    def test_get_fraction_of_outgoing_interactions_stats(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_cdr_swapped = swap_caller_and_recipient(spark_cdr_with_weekend)
        spark_fraction_of_outgoing_interactions = (
            get_outgoing_interaction_fraction_stats(spark_cdr_swapped)
        )
        pd_cdr_fraction_of_outgoing_interactions = (
            spark_fraction_of_outgoing_interactions.toPandas()
        )

        assert pd_cdr_fraction_of_outgoing_interactions.shape == (4, 57)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_text_mean_fraction_of_outgoing_calls",
                "weekday_nighttime_call_mean_fraction_of_outgoing_calls",
                "weekend_nighttime_text_mean_fraction_of_outgoing_calls",
                "weekend_nighttime_call_mean_fraction_of_outgoing_calls",
                "weekday_daytime_text_mean_fraction_of_outgoing_calls",
                "weekday_daytime_call_mean_fraction_of_outgoing_calls",
                "weekend_daytime_text_mean_fraction_of_outgoing_calls",
                "weekend_daytime_call_mean_fraction_of_outgoing_calls",
                "weekday_nighttime_text_min_fraction_of_outgoing_calls",
                "weekday_nighttime_call_min_fraction_of_outgoing_calls",
                "weekend_nighttime_text_min_fraction_of_outgoing_calls",
                "weekend_nighttime_call_min_fraction_of_outgoing_calls",
                "weekday_daytime_text_min_fraction_of_outgoing_calls",
                "weekday_daytime_call_min_fraction_of_outgoing_calls",
                "weekend_daytime_text_min_fraction_of_outgoing_calls",
                "weekend_daytime_call_min_fraction_of_outgoing_calls",
                "weekday_nighttime_text_max_fraction_of_outgoing_calls",
                "weekday_nighttime_call_max_fraction_of_outgoing_calls",
                "weekend_nighttime_text_max_fraction_of_outgoing_calls",
                "weekend_nighttime_call_max_fraction_of_outgoing_calls",
                "weekday_daytime_text_max_fraction_of_outgoing_calls",
                "weekday_daytime_call_max_fraction_of_outgoing_calls",
                "weekend_daytime_text_max_fraction_of_outgoing_calls",
                "weekend_daytime_call_max_fraction_of_outgoing_calls",
                "weekday_nighttime_text_std_fraction_of_outgoing_calls",
                "weekday_nighttime_call_std_fraction_of_outgoing_calls",
                "weekend_nighttime_text_std_fraction_of_outgoing_calls",
                "weekend_nighttime_call_std_fraction_of_outgoing_calls",
                "weekday_daytime_text_std_fraction_of_outgoing_calls",
                "weekday_daytime_call_std_fraction_of_outgoing_calls",
                "weekend_daytime_text_std_fraction_of_outgoing_calls",
                "weekend_daytime_call_std_fraction_of_outgoing_calls",
                "weekday_nighttime_text_median_fraction_of_outgoing_calls",
                "weekday_nighttime_call_median_fraction_of_outgoing_calls",
                "weekend_nighttime_text_median_fraction_of_outgoing_calls",
                "weekend_nighttime_call_median_fraction_of_outgoing_calls",
                "weekday_daytime_text_median_fraction_of_outgoing_calls",
                "weekday_daytime_call_median_fraction_of_outgoing_calls",
                "weekend_daytime_text_median_fraction_of_outgoing_calls",
                "weekend_daytime_call_median_fraction_of_outgoing_calls",
                "weekday_nighttime_text_skewness_fraction_of_outgoing_calls",
                "weekday_nighttime_call_skewness_fraction_of_outgoing_calls",
                "weekend_nighttime_text_skewness_fraction_of_outgoing_calls",
                "weekend_nighttime_call_skewness_fraction_of_outgoing_calls",
                "weekday_daytime_text_skewness_fraction_of_outgoing_calls",
                "weekday_daytime_call_skewness_fraction_of_outgoing_calls",
                "weekend_daytime_text_skewness_fraction_of_outgoing_calls",
                "weekend_daytime_call_skewness_fraction_of_outgoing_calls",
                "weekday_nighttime_text_kurtosis_fraction_of_outgoing_calls",
                "weekday_nighttime_call_kurtosis_fraction_of_outgoing_calls",
                "weekend_nighttime_text_kurtosis_fraction_of_outgoing_calls",
                "weekend_nighttime_call_kurtosis_fraction_of_outgoing_calls",
                "weekday_daytime_text_kurtosis_fraction_of_outgoing_calls",
                "weekday_daytime_call_kurtosis_fraction_of_outgoing_calls",
                "weekend_daytime_text_kurtosis_fraction_of_outgoing_calls",
                "weekend_daytime_call_kurtosis_fraction_of_outgoing_calls",
            ]
        ) == set(pd_cdr_fraction_of_outgoing_interactions.columns)

        pd_cdr_swapped = spark_cdr_swapped.toPandas()
        for col in [
            "caller_id",
            "recipient_id",
            "is_weekend",
            "is_daytime",
            "transaction_type",
            "direction_of_transaction",
        ]:
            spark_cdr_no_col = spark.createDataFrame(pd_cdr_swapped.drop(columns=[col]))
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'recipient_id', 'is_weekend', 'is_daytime', 'direction_of_transaction' and 'transaction_type' columns",
            ):
                get_outgoing_interaction_fraction_stats(spark_cdr_no_col)

    def test_get_interaction_stats_per_caller(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_cdr_swapped = swap_caller_and_recipient(spark_cdr_with_weekend)
        spark_interaction_stats_per_caller = get_interaction_stats_per_caller(
            spark_cdr_swapped
        )
        pd_cdr_interaction_stats = spark_interaction_stats_per_caller.toPandas()

        assert pd_cdr_interaction_stats.shape == (4, 57)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_text_mean_interaction_count",
                "weekday_nighttime_call_mean_interaction_count",
                "weekend_nighttime_text_mean_interaction_count",
                "weekend_nighttime_call_mean_interaction_count",
                "weekday_daytime_text_mean_interaction_count",
                "weekday_daytime_call_mean_interaction_count",
                "weekend_daytime_text_mean_interaction_count",
                "weekend_daytime_call_mean_interaction_count",
                "weekday_nighttime_text_min_interaction_count",
                "weekday_nighttime_call_min_interaction_count",
                "weekend_nighttime_text_min_interaction_count",
                "weekend_nighttime_call_min_interaction_count",
                "weekday_daytime_text_min_interaction_count",
                "weekday_daytime_call_min_interaction_count",
                "weekend_daytime_text_min_interaction_count",
                "weekend_daytime_call_min_interaction_count",
                "weekday_nighttime_text_max_interaction_count",
                "weekday_nighttime_call_max_interaction_count",
                "weekend_nighttime_text_max_interaction_count",
                "weekend_nighttime_call_max_interaction_count",
                "weekday_daytime_text_max_interaction_count",
                "weekday_daytime_call_max_interaction_count",
                "weekend_daytime_text_max_interaction_count",
                "weekend_daytime_call_max_interaction_count",
                "weekday_nighttime_text_std_interaction_count",
                "weekday_nighttime_call_std_interaction_count",
                "weekend_nighttime_text_std_interaction_count",
                "weekend_nighttime_call_std_interaction_count",
                "weekday_daytime_text_std_interaction_count",
                "weekday_daytime_call_std_interaction_count",
                "weekend_daytime_text_std_interaction_count",
                "weekend_daytime_call_std_interaction_count",
                "weekday_nighttime_text_median_interaction_count",
                "weekday_nighttime_call_median_interaction_count",
                "weekend_nighttime_text_median_interaction_count",
                "weekend_nighttime_call_median_interaction_count",
                "weekday_daytime_text_median_interaction_count",
                "weekday_daytime_call_median_interaction_count",
                "weekend_daytime_text_median_interaction_count",
                "weekend_daytime_call_median_interaction_count",
                "weekday_nighttime_text_skewness_interaction_count",
                "weekday_nighttime_call_skewness_interaction_count",
                "weekend_nighttime_text_skewness_interaction_count",
                "weekend_nighttime_call_skewness_interaction_count",
                "weekday_daytime_text_skewness_interaction_count",
                "weekday_daytime_call_skewness_interaction_count",
                "weekend_daytime_text_skewness_interaction_count",
                "weekend_daytime_call_skewness_interaction_count",
                "weekday_nighttime_text_kurtosis_interaction_count",
                "weekday_nighttime_call_kurtosis_interaction_count",
                "weekend_nighttime_text_kurtosis_interaction_count",
                "weekend_nighttime_call_kurtosis_interaction_count",
                "weekday_daytime_text_kurtosis_interaction_count",
                "weekday_daytime_call_kurtosis_interaction_count",
                "weekend_daytime_text_kurtosis_interaction_count",
                "weekend_daytime_call_kurtosis_interaction_count",
            ]
        ) == set(pd_cdr_interaction_stats.columns)

        pd_cdr_with_weekend = spark_cdr_with_weekend.toPandas()
        for col in [
            "caller_id",
            "recipient_id",
            "is_weekend",
            "is_daytime",
            "transaction_type",
        ]:
            spark_cdr_no_col = spark.createDataFrame(
                pd_cdr_with_weekend.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'recipient_id', 'is_weekend', 'is_daytime', and 'transaction_type' columns",
            ):
                get_interaction_stats_per_caller(spark_cdr_no_col)

    def test_get_inter_event_time_stats(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_inter_event_time_stats = get_inter_event_time_stats(
            spark_cdr_with_weekend
        )
        pd_cdr_inter_event_time_stats = spark_inter_event_time_stats.toPandas()

        assert pd_cdr_inter_event_time_stats.shape == (3, 57)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_text_mean_inter_event_time",
                "weekday_nighttime_call_mean_inter_event_time",
                "weekend_nighttime_text_mean_inter_event_time",
                "weekend_nighttime_call_mean_inter_event_time",
                "weekday_daytime_text_mean_inter_event_time",
                "weekday_daytime_call_mean_inter_event_time",
                "weekend_daytime_text_mean_inter_event_time",
                "weekend_daytime_call_mean_inter_event_time",
                "weekday_nighttime_text_min_inter_event_time",
                "weekday_nighttime_call_min_inter_event_time",
                "weekend_nighttime_text_min_inter_event_time",
                "weekend_nighttime_call_min_inter_event_time",
                "weekday_daytime_text_min_inter_event_time",
                "weekday_daytime_call_min_inter_event_time",
                "weekend_daytime_text_min_inter_event_time",
                "weekend_daytime_call_min_inter_event_time",
                "weekday_nighttime_text_max_inter_event_time",
                "weekday_nighttime_call_max_inter_event_time",
                "weekend_nighttime_text_max_inter_event_time",
                "weekend_nighttime_call_max_inter_event_time",
                "weekday_daytime_text_max_inter_event_time",
                "weekday_daytime_call_max_inter_event_time",
                "weekend_daytime_text_max_inter_event_time",
                "weekend_daytime_call_max_inter_event_time",
                "weekday_nighttime_text_std_inter_event_time",
                "weekday_nighttime_call_std_inter_event_time",
                "weekend_nighttime_text_std_inter_event_time",
                "weekend_nighttime_call_std_inter_event_time",
                "weekday_daytime_text_std_inter_event_time",
                "weekday_daytime_call_std_inter_event_time",
                "weekend_daytime_text_std_inter_event_time",
                "weekend_daytime_call_std_inter_event_time",
                "weekday_nighttime_text_median_inter_event_time",
                "weekday_nighttime_call_median_inter_event_time",
                "weekend_nighttime_text_median_inter_event_time",
                "weekend_nighttime_call_median_inter_event_time",
                "weekday_daytime_text_median_inter_event_time",
                "weekday_daytime_call_median_inter_event_time",
                "weekend_daytime_text_median_inter_event_time",
                "weekend_daytime_call_median_inter_event_time",
                "weekday_nighttime_text_skewness_inter_event_time",
                "weekday_nighttime_call_skewness_inter_event_time",
                "weekend_nighttime_text_skewness_inter_event_time",
                "weekend_nighttime_call_skewness_inter_event_time",
                "weekday_daytime_text_skewness_inter_event_time",
                "weekday_daytime_call_skewness_inter_event_time",
                "weekend_daytime_text_skewness_inter_event_time",
                "weekend_daytime_call_skewness_inter_event_time",
                "weekday_nighttime_text_kurtosis_inter_event_time",
                "weekday_nighttime_call_kurtosis_inter_event_time",
                "weekend_nighttime_text_kurtosis_inter_event_time",
                "weekend_nighttime_call_kurtosis_inter_event_time",
                "weekday_daytime_text_kurtosis_inter_event_time",
                "weekday_daytime_call_kurtosis_inter_event_time",
                "weekend_daytime_text_kurtosis_inter_event_time",
                "weekend_daytime_call_kurtosis_inter_event_time",
            ]
        ) == set(pd_cdr_inter_event_time_stats.columns)

        pd_cdr_with_weekend = spark_cdr_with_weekend.toPandas()
        for col in [
            "caller_id",
            "is_weekend",
            "is_daytime",
            "transaction_type",
            "timestamp",
        ]:
            spark_cdr_no_col = spark.createDataFrame(
                pd_cdr_with_weekend.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'timestamp', 'is_weekend', 'is_daytime', and 'transaction_type' columns",
            ):
                get_inter_event_time_stats(spark_cdr_no_col)

    def test_get_pareto_principle_interaction_stats(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_pareto_stats = get_pareto_principle_interaction_stats(
            spark_cdr_with_weekend
        )
        pd_cdr_pareto_stats = spark_pareto_stats.toPandas()

        assert pd_cdr_pareto_stats.shape == (3, 9)
        assert pd_cdr_pareto_stats.filter(
            like="pareto_principle_interaction_fraction"
        ).sum(1).tolist() == pytest.approx([1.0, 1.0, 1.0], rel=1e-2)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_text_pareto_principle_interaction_fraction",
                "weekday_nighttime_call_pareto_principle_interaction_fraction",
                "weekend_nighttime_text_pareto_principle_interaction_fraction",
                "weekend_nighttime_call_pareto_principle_interaction_fraction",
                "weekday_daytime_text_pareto_principle_interaction_fraction",
                "weekday_daytime_call_pareto_principle_interaction_fraction",
                "weekend_daytime_text_pareto_principle_interaction_fraction",
                "weekend_daytime_call_pareto_principle_interaction_fraction",
            ]
        ) == set(pd_cdr_pareto_stats.columns)

        pd_cdr_with_weekend = spark_cdr_with_weekend.toPandas()
        for col in [
            "caller_id",
            "recipient_id",
            "is_weekend",
            "is_daytime",
            "transaction_type",
        ]:
            spark_cdr_no_col = spark.createDataFrame(
                pd_cdr_with_weekend.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'recipient_id', 'is_weekend', 'is_daytime', and 'transaction_type' columns",
            ):
                get_pareto_principle_interaction_stats(spark_cdr_no_col)

    def test_get_pareto_principle_call_duration_stats(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_pareto_call_stats = get_pareto_principle_call_duration_stats(
            spark_cdr_with_weekend
        )
        pd_cdr_pareto_call_stats = spark_pareto_call_stats.toPandas()

        assert pd_cdr_pareto_call_stats.shape == (3, 5)
        assert pd_cdr_pareto_call_stats.filter(
            like="pareto_call_duration_fraction"
        ).sum(1).tolist() == pytest.approx([1.0, 1.0, 1.0], rel=1e-2)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_pareto_call_duration_fraction",
                "weekend_nighttime_pareto_call_duration_fraction",
                "weekday_daytime_pareto_call_duration_fraction",
                "weekend_daytime_pareto_call_duration_fraction",
            ]
        ) == set(pd_cdr_pareto_call_stats.columns)

        pd_cdr_with_weekend = spark_cdr_with_weekend.toPandas()
        for col in [
            "caller_id",
            "recipient_id",
            "is_weekend",
            "is_daytime",
            "transaction_type",
        ]:
            spark_cdr_no_col = spark.createDataFrame(
                pd_cdr_with_weekend.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'recipient_id', 'is_weekend', 'is_daytime', 'transaction_type', and 'duration' columns",
            ):
                get_pareto_principle_call_duration_stats(spark_cdr_no_col)

    def test_get_number_of_interactions_per_user(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_cdr_swapped = swap_caller_and_recipient(spark_cdr_with_weekend)

        spark_number_of_interactions = get_number_of_interactions_per_user(
            spark_cdr_swapped
        )
        pd_cdr_number_of_interactions = spark_number_of_interactions.toPandas()
        assert pd_cdr_number_of_interactions.shape == (4, 17)
        assert pd_cdr_number_of_interactions.filter(like="num_interactions").sum(
            1
        ).tolist() == [1, 1, 1, 1]
        assert set(
            [
                "caller_id",
                "weekday_nighttime_text_incoming_num_interactions",
                "weekday_nighttime_call_incoming_num_interactions",
                "weekend_nighttime_text_incoming_num_interactions",
                "weekend_nighttime_call_incoming_num_interactions",
                "weekday_daytime_text_incoming_num_interactions",
                "weekday_daytime_call_incoming_num_interactions",
                "weekend_daytime_text_incoming_num_interactions",
                "weekend_daytime_call_incoming_num_interactions",
                "weekday_nighttime_text_outgoing_num_interactions",
                "weekday_nighttime_call_outgoing_num_interactions",
                "weekend_nighttime_text_outgoing_num_interactions",
                "weekend_nighttime_call_outgoing_num_interactions",
                "weekday_daytime_text_outgoing_num_interactions",
                "weekday_daytime_call_outgoing_num_interactions",
                "weekend_daytime_text_outgoing_num_interactions",
                "weekend_daytime_call_outgoing_num_interactions",
            ]
        ) == set(pd_cdr_number_of_interactions.columns)

        pd_cdr_swapped = spark_cdr_swapped.toPandas()
        for col in [
            "caller_id",
            "is_weekend",
            "is_daytime",
            "transaction_type",
            "direction_of_transaction",
        ]:
            spark_cdr_no_col = spark.createDataFrame(pd_cdr_swapped.drop(columns=[col]))
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'is_weekend', 'is_daytime', 'transaction_type', and 'direction_of_transaction' columns",
            ):
                get_number_of_interactions_per_user(spark_cdr_no_col)

    def test_get_number_of_antennas(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_cdr_swapped = swap_caller_and_recipient(spark_cdr_with_weekend)

        spark_number_of_antennas = get_number_of_antennas(spark_cdr_swapped)
        pd_cdr_number_of_antennas = spark_number_of_antennas.toPandas()
        assert pd_cdr_number_of_antennas.shape == (4, 5)
        assert pd_cdr_number_of_antennas.filter(like="num_unique_antennas").sum(
            1
        ).tolist() == [2, 1, 1, 1]
        assert set(
            [
                "caller_id",
                "weekday_nighttime_num_unique_antennas",
                "weekend_nighttime_num_unique_antennas",
                "weekday_daytime_num_unique_antennas",
                "weekend_daytime_num_unique_antennas",
            ]
        ) == set(pd_cdr_number_of_antennas.columns)

        pd_cdr_swapped = spark_cdr_swapped.toPandas()
        for col in ["caller_id", "caller_antenna_id", "is_daytime", "is_weekend"]:
            spark_cdr_no_col = spark.createDataFrame(pd_cdr_swapped.drop(columns=[col]))
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'caller_antenna_id', 'is_daytime', and 'is_weekend' columns",
            ):
                get_number_of_antennas(spark_cdr_no_col)

    def test_get_entropy_of_antennas_per_caller(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)

        spark_entropy_of_antennas = get_entropy_of_antennas_per_caller(
            spark_cdr_with_weekend
        )
        pd_cdr_entropy_of_antennas = spark_entropy_of_antennas.toPandas()
        assert pd_cdr_entropy_of_antennas.shape == (3, 5)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_entropy_of_antennas",
                "weekend_nighttime_entropy_of_antennas",
                "weekday_daytime_entropy_of_antennas",
                "weekend_daytime_entropy_of_antennas",
            ]
        ) == set(pd_cdr_entropy_of_antennas.columns)

        pd_cdr_with_weekend = spark_cdr_with_weekend.toPandas()
        for col in [
            "caller_id",
            "caller_antenna_id",
            "is_weekend",
            "is_daytime",
        ]:
            spark_cdr_no_col = spark.createDataFrame(
                pd_cdr_with_weekend.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'caller_antenna_id', 'is_daytime', and 'is_weekend' columns",
            ):
                get_entropy_of_antennas_per_caller(spark_cdr_no_col)

    def test_get_radius_of_gyration(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)
        pd_antenna_data = pd.DataFrame(ANTENNA_DATA)
        pd_antenna_data.rename(
            columns={"antenna_id": "caller_antenna_id"}, inplace=True
        )

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)

        spark_antenna_data = spark.createDataFrame(pd_antenna_data)

        spark_radius_of_gyration = get_radius_of_gyration(
            spark_cdr_with_weekend, spark_antenna_data
        )
        pd_cdr_radius_of_gyration = spark_radius_of_gyration.toPandas()
        assert pd_cdr_radius_of_gyration.shape == (3, 5)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_radius_of_gyration",
                "weekend_nighttime_radius_of_gyration",
                "weekday_daytime_radius_of_gyration",
                "weekend_daytime_radius_of_gyration",
            ]
        ) == set(pd_cdr_radius_of_gyration.columns)

        pd_cdr_with_weekend = spark_cdr_with_weekend.toPandas()
        for col in [
            "caller_id",
            "caller_antenna_id",
            "is_weekend",
            "is_daytime",
        ]:
            spark_cdr_no_col = spark.createDataFrame(
                pd_cdr_with_weekend.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'caller_antenna_id', 'is_weekend', and 'is_daytime' columns",
            ):
                get_radius_of_gyration(spark_cdr_no_col, spark_antenna_data)

        for col in [
            "caller_antenna_id",
            "latitude",
            "longitude",
        ]:
            spark_antenna_no_col = spark.createDataFrame(
                pd_antenna_data.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Antennas dataframe must contain 'caller_antenna_id', 'latitude', and 'longitude' columns",
            ):
                get_radius_of_gyration(spark_cdr_with_weekend, spark_antenna_no_col)

    def test_get_pareto_principle_antennas(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)

        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)

        spark_pareto_antennas = get_pareto_principle_antennas(
            spark_cdr_with_weekend, percentage_threshold=0.8
        )
        pd_cdr_pareto_antennas = spark_pareto_antennas.toPandas()
        assert pd_cdr_pareto_antennas.shape == (3, 5)
        assert pd_cdr_pareto_antennas.filter(like="num_pareto_principle_antennas").sum(
            1
        ).tolist() == [2, 1, 1]
        assert set(
            [
                "caller_id",
                "weekday_nighttime_num_pareto_principle_antennas",
                "weekend_nighttime_num_pareto_principle_antennas",
                "weekday_daytime_num_pareto_principle_antennas",
                "weekend_daytime_num_pareto_principle_antennas",
            ]
        ) == set(pd_cdr_pareto_antennas.columns)

        pd_cdr_with_weekend = spark_cdr_with_weekend.toPandas()
        for col in [
            "caller_id",
            "caller_antenna_id",
            "is_weekend",
            "is_daytime",
        ]:
            spark_cdr_no_col = spark.createDataFrame(
                pd_cdr_with_weekend.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'caller_antenna_id', 'is_daytime', and 'is_weekend' columns",
            ):
                get_pareto_principle_antennas(spark_cdr_no_col)

    def test_get_average_num_of_interactions_from_home_antenna(self, spark):
        pd_cdr_data = pd.DataFrame(CDR_DATA)
        spark_cdr_data = spark.createDataFrame(pd_cdr_data)
        spark_cdr_with_daytime = identify_daytime(spark_cdr_data)
        spark_cdr_with_weekend = identify_weekend(spark_cdr_with_daytime)
        spark_avg_home_antenna_interactions = (
            get_average_num_of_interactions_from_home_antennas(spark_cdr_with_weekend)
        )

        pd_avg_home_antenna_interactions = (
            spark_avg_home_antenna_interactions.toPandas()
        )

        assert pd_avg_home_antenna_interactions.shape == (1, 5)
        assert set(
            [
                "caller_id",
                "weekday_nighttime_mean_home_antenna_interaction",
                "weekend_nighttime_mean_home_antenna_interaction",
                "weekday_daytime_mean_home_antenna_interaction",
                "weekend_daytime_mean_home_antenna_interaction",
            ]
        ) == set(pd_avg_home_antenna_interactions.columns)
        assert pd_avg_home_antenna_interactions.filter(
            like="mean_home_antenna_interaction"
        ).sum(1).tolist() == [0.0]

        pd_cdr_with_weekend = spark_cdr_with_weekend.toPandas()
        for col in ["caller_id", "caller_antenna_id", "is_weekend", "is_daytime"]:
            spark_cdr_no_col = spark.createDataFrame(
                pd_cdr_with_weekend.drop(columns=[col])
            )
            with pytest.raises(
                ValueError,
                match="Dataframe must contain 'caller_id', 'caller_antenna_id', 'is_daytime', and 'is_weekend' columns",
            ):
                get_average_num_of_interactions_from_home_antennas(spark_cdr_no_col)
