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

import pytest
import pandas as pd
import numpy as np
from cider.validation_metrics.dependencies import (
    calculate_weighted_spearmanr,
    calculate_weighted_pearsonr,
    convert_threshold_to_percentile,
    calculate_metrics_binary_valued_consumption,
    calculate_utility,
    where_is_false_positive_rate_nonmonotonic,
    calculate_rank_residuals_by_characteristic,
    calculate_demographic_parity_per_characteristic,
    calculate_independence_btwn_proxy_and_characteristic,
    calculate_precision_and_recall_independence_characteristic,
)
from cider.validation_metrics.schemas import (
    ConsumptionData,
    ConsumptionColumn,
    ConsumptionDataWithCharacteristic,
)
from cider.validation_metrics.core import (
    compute_auc_roc_precision_recall_with_percentile_grid,
    compute_utility_grid,
    calculate_optimal_utility_and_cash_transfer_size_table,
    calculate_rank_residuals_table_by_characteristic,
    calculate_demographic_parity_table_per_characteristic,
    combine_tables_on_characteristic,
)
from conftest import (
    HOUSEHOLD_CONSUMPTION_DATA,
    HOUSEHOLD_CONSUMPTION_DATA_W_CHARACTERISTIC,
)


class TestValidationMetricsDependencies:

    household_consumption_data = pd.DataFrame(HOUSEHOLD_CONSUMPTION_DATA)
    household_consumption_data_w_characteristic = pd.DataFrame(
        HOUSEHOLD_CONSUMPTION_DATA_W_CHARACTERISTIC
    )

    def test_missing_columns_raise_errors_for_consumption_data(self):
        for col in ConsumptionData.model_fields.keys():
            household_data_no_cols = self.household_consumption_data.drop(columns=[col])
            with pytest.raises(ValueError):
                convert_threshold_to_percentile(50.0, household_data_no_cols)
            with pytest.raises(ValueError):
                calculate_weighted_spearmanr(household_data_no_cols)
            with pytest.raises(ValueError):
                calculate_weighted_pearsonr(household_data_no_cols)
            with pytest.raises(ValueError):
                calculate_metrics_binary_valued_consumption(
                    household_data_no_cols, 50.0, 50.0
                )
            with pytest.raises(ValueError):
                calculate_utility(
                    household_data_no_cols, 50.0, ConsumptionColumn.GROUNDTRUTH, 1000
                )

    def test_missing_columns_raise_errors_for_consumption_data_with_characteristic(
        self,
    ):
        for col in ConsumptionDataWithCharacteristic.model_fields.keys():
            household_data_no_cols = (
                self.household_consumption_data_w_characteristic.drop(columns=[col])
            )
            with pytest.raises(ValueError):
                calculate_rank_residuals_by_characteristic(household_data_no_cols)

            with pytest.raises(ValueError):
                calculate_demographic_parity_per_characteristic(
                    household_data_no_cols, threshold_percentile=50
                )

            with pytest.raises(ValueError):
                calculate_independence_btwn_proxy_and_characteristic(
                    household_data_no_cols, threshold_percentile=50
                )

            with pytest.raises(ValueError):
                calculate_precision_and_recall_independence_characteristic(
                    household_data_no_cols, 50, 50
                )

    @pytest.mark.parametrize(
        "threshold,expected_percentile",
        [
            (5.0, 77.848),
            (2.0, 31.285),
            (1.0, 28.98),
        ],
    )
    def test_convert_threshold_to_percentile(self, threshold, expected_percentile):
        percentile = convert_threshold_to_percentile(
            threshold, self.household_consumption_data
        )
        assert pytest.approx(percentile, 1e-3) == expected_percentile

    @pytest.mark.parametrize(
        "significant_digits,expected_spearmanr",
        [
            (2, 0.9),
            (3, 0.896),
            (4, 0.8956),
        ],
    )
    def test_calculate_weighted_spearmanr(self, significant_digits, expected_spearmanr):
        spearmanr = calculate_weighted_spearmanr(
            self.household_consumption_data, significant_digits=significant_digits
        )
        assert (
            pytest.approx(spearmanr, 10 ** (-significant_digits)) == expected_spearmanr
        )

    @pytest.mark.parametrize(
        "significant_digits,expected_pearsonr",
        [
            (2, 0.94),
            (3, 0.941),
            (4, 0.9409),
        ],
    )
    def test_calculate_weighted_pearsonr(self, significant_digits, expected_pearsonr):
        pearsonr = calculate_weighted_pearsonr(
            self.household_consumption_data, significant_digits=significant_digits
        )
        assert pytest.approx(pearsonr, 10 ** (-significant_digits)) == expected_pearsonr

    @pytest.mark.parametrize(
        "groundtruth_threshold_percentile,proxy_threshold_percentile,expected_accuracy,expected_precision,expected_recall,expected_tpr,expected_fpr,expected_auc,expected_roc_curve, expected_spearman_r, expected_pearson_r",
        [
            (
                25,
                25,
                0.727,
                0.6,
                0.75,
                0.75,
                0.286,
                0.857,
                (
                    [0.0, 0.0, 0.571, 0.571, 0.714, 1.0],
                    [0.0, 0.75, 0.75, 1.0, 1.0, 1.0],
                ),
                0.9,
                0.94,
            ),
            (
                25,
                50,
                0.545,
                0.429,
                0.75,
                0.75,
                0.571,
                0.857,
                (
                    [0.0, 0.0, 0.571, 0.571, 0.714, 1.0],
                    [0.0, 0.75, 0.75, 1.0, 1.0, 1.0],
                ),
                0.9,
                0.94,
            ),
            (
                50,
                25,
                0.909,
                1.0,
                0.833,
                0.833,
                0.0,
                0.933,
                (
                    [0.0, 0.0, 0.0, 0.4, 0.4, 0.6, 1.0],
                    [0.0, 0.5, 0.833, 0.833, 1.0, 1.0, 1.0],
                ),
                0.9,
                0.94,
            ),
        ],
    )
    def test_calculate_metrics_binary_valued_consumption(
        self,
        groundtruth_threshold_percentile,
        proxy_threshold_percentile,
        expected_accuracy,
        expected_precision,
        expected_recall,
        expected_tpr,
        expected_fpr,
        expected_auc,
        expected_roc_curve,
        expected_spearman_r,
        expected_pearson_r,
    ):
        results = calculate_metrics_binary_valued_consumption(
            self.household_consumption_data,
            groundtruth_threshold_percentile,
            proxy_threshold_percentile,
        )

        assert pytest.approx(results["accuracy"], 1e-2) == expected_accuracy
        assert pytest.approx(results["precision"], 1e-2) == expected_precision
        assert pytest.approx(results["recall"], 1e-2) == expected_recall
        assert pytest.approx(results["true_positive_rate"], 1e-2) == expected_tpr
        assert pytest.approx(results["false_positive_rate"], 1e-2) == expected_fpr
        assert pytest.approx(results["auc"], 1e-2) == expected_auc
        roc_curve_fpr, roc_curve_tpr, _ = results["roc_curve"].to_numpy()[0]
        assert pytest.approx(roc_curve_fpr, 1e-2) == expected_roc_curve[0]
        assert pytest.approx(roc_curve_tpr, 1e-2) == expected_roc_curve[1]
        assert pytest.approx(results["spearman_r"], 1e-2) == expected_spearman_r
        assert pytest.approx(results["pearson_r"], 1e-2) == expected_pearson_r

    @pytest.mark.parametrize(
        "consumption_column,threshold_percentile,expected_utility",
        [
            (ConsumptionColumn.GROUNDTRUTH, 20.0, -0.1115),
            (ConsumptionColumn.PROXY, 30.0, -0.0378),
        ],
    )
    def test_calculate_utility(
        self, consumption_column, threshold_percentile, expected_utility
    ):
        utility = calculate_utility(
            self.household_consumption_data,
            threshold_percentile=threshold_percentile,
            consumption_column=consumption_column,
            cash_transfer_amount=1000,
        )
        assert pytest.approx(utility, 1e-2) == expected_utility

    def test_is_false_positive_rate_monotonic(self):
        # Test with a monotonic false positive rate series
        fpr_monotonic = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        assert len(where_is_false_positive_rate_nonmonotonic(fpr_monotonic)) == 5

        # Test with a non-monotonic false positive rate series
        fpr_non_monotonic = np.array([0.0, 0.2, 0.15, 0.3, 0.4, 0.5])
        assert len(where_is_false_positive_rate_nonmonotonic(fpr_non_monotonic)) == 4

    def test_calculate_rank_residuals_by_characteristic(self):
        rank_residuals = calculate_rank_residuals_by_characteristic(
            self.household_consumption_data_w_characteristic
        )
        rank_residuals_list = rank_residuals.to_dict()
        pytest.approx(rank_residuals_list["group_1"], 1e-2) == [-1, 1, -1, 1]
        pytest.approx(rank_residuals_list["group_2"], 1e-2) == [1, -1, 1, -1]

    @pytest.mark.parametrize(
        "threshold_percentile,expected_demographic_parity",
        [
            (50, 1.0),
            (25, 0.5),
            (10, 0.0),
        ],
    )
    def test_calculate_demographic_parity_per_characteristic(
        self, threshold_percentile, expected_demographic_parity
    ):
        results = calculate_demographic_parity_per_characteristic(
            self.household_consumption_data_w_characteristic,
            threshold_percentile=threshold_percentile,
        )
        assert set(
            [
                "groundtruth_poverty_percentage",
                "proxy_poverty_percentage",
                "demographic_parity",
            ]
        ) == set(results.columns)
        pytest.approx(results.loc["group_1", "demographic_parity"], 1e-2) == [
            expected_demographic_parity
        ]
        pytest.approx(results.loc["group_2", "demographic_parity"], 1e-2) == [
            expected_demographic_parity
        ]

    @pytest.mark.parametrize(
        "threshold_percentile,expected_independence_p_value",
        [
            (50, (0.0035, 0.9528)),
            (25, (2.753, 0.097)),
            (10, (0.692, 0.406)),
        ],
    )
    def test_calculate_independence_btwn_proxy_and_characteristic(
        self, threshold_percentile, expected_independence_p_value
    ):
        results_df = calculate_independence_btwn_proxy_and_characteristic(
            self.household_consumption_data_w_characteristic,
            threshold_percentile=threshold_percentile,
        )
        assert (
            pytest.approx(results_df["chi2_statistic"][0], 1e-2)
            == expected_independence_p_value[0]
        )
        assert (
            pytest.approx(results_df["p_value"][0], 1e-2)
            == expected_independence_p_value[1]
        )

    @pytest.mark.parametrize(
        "groundtruth_threshold_percentile,proxy_threshold_percentile,expected_precision_chi2,expected_precision_p_value,expected_recall_chi2,expected_recall_p_value",
        [
            (50, 50, 2.9575, 0.0855, 0.96, 0.3272),
            (25, 50, 0.3646, 0.546, 0.4444, 0.505),
            (10, 25, 0.0, 1.0, 0.0, 1.0),
        ],
    )
    def test_calculate_precision_and_recall_independence_characteristic(
        self,
        groundtruth_threshold_percentile,
        proxy_threshold_percentile,
        expected_precision_chi2,
        expected_precision_p_value,
        expected_recall_chi2,
        expected_recall_p_value,
    ):
        results_df = calculate_precision_and_recall_independence_characteristic(
            self.household_consumption_data_w_characteristic,
            groundtruth_threshold_percentile,
            proxy_threshold_percentile,
        )
        assert (
            pytest.approx(results_df.loc["precision", "chi2_statistic"], 1e-2)
            == expected_precision_chi2
        )
        assert (
            pytest.approx(results_df.loc["precision", "p_value"], 1e-2)
            == expected_precision_p_value
        )
        assert (
            pytest.approx(results_df.loc["recall", "chi2_statistic"], 1e-2)
            == expected_recall_chi2
        )
        assert (
            pytest.approx(results_df.loc["recall", "p_value"], 1e-2)
            == expected_recall_p_value
        )


class TestValidationMetricsCore:

    household_consumption_data = pd.DataFrame(HOUSEHOLD_CONSUMPTION_DATA)
    household_consumption_data_w_characteristic = pd.DataFrame(
        HOUSEHOLD_CONSUMPTION_DATA_W_CHARACTERISTIC
    )

    def test_missing_columns_raise_errors(self):
        for col in ConsumptionData.model_fields.keys():
            household_data_no_cols = self.household_consumption_data.drop(columns=[col])
            with pytest.raises(ValueError):
                compute_auc_roc_precision_recall_with_percentile_grid(
                    household_data_no_cols,
                    fixed_groundtruth_percentile=20,
                    num_grid_points=10,
                )
            with pytest.raises(ValueError):
                compute_utility_grid(
                    household_data_no_cols,
                    cash_transfer_amount=1000,
                    num_grid_points=10,
                    constant_relative_risk_aversion=3.0,
                )
            with pytest.raises(ValueError):
                calculate_optimal_utility_and_cash_transfer_size_table(
                    household_data_no_cols,
                    cash_transfer_amount=1000,
                    num_grid_points=10,
                    constant_relative_risk_aversion=3.0,
                )

    def test_missing_columns_raise_errors_with_characteristic(self):
        for col in ConsumptionDataWithCharacteristic.model_fields.keys():
            household_data_no_cols = (
                self.household_consumption_data_w_characteristic.drop(columns=[col])
            )
            with pytest.raises(ValueError):
                calculate_rank_residuals_table_by_characteristic(household_data_no_cols)
            with pytest.raises(ValueError):
                calculate_demographic_parity_table_per_characteristic(
                    household_data_no_cols, threshold_percentile=50
                )
            with pytest.raises(ValueError):
                combine_tables_on_characteristic(
                    household_data_no_cols, threshold_percentile=50
                )

    def test_compute_auc_roc_with_percentile_grid(self):
        results_df = compute_auc_roc_precision_recall_with_percentile_grid(
            self.household_consumption_data,
            fixed_groundtruth_percentile=20,
            num_grid_points=10,
        )
        assert set(
            [
                "percentile",
                "precision",
                "recall",
                "true_positive_rate",
                "false_positive_rate",
                "auc",
            ]
        ) == set(results_df.columns)
        assert len(results_df) == 10

    def test_compute_utility_grid(self):
        results_df = compute_utility_grid(
            self.household_consumption_data,
            cash_transfer_amount=1000,
            num_grid_points=10,
            constant_relative_risk_aversion=3.0,
        )
        assert set(
            [
                "percentile",
                "cash_transfer_amount_groundtruth",
                "utility_groundtruth",
                "cash_transfer_amount_proxy",
                "utility_proxy",
            ]
        ) == set(results_df.columns)
        assert len(results_df) == 10

    def test_calculate_optimal_utility_and_cash_transfer_size_table(self):
        results_df = calculate_optimal_utility_and_cash_transfer_size_table(
            self.household_consumption_data,
            cash_transfer_amount=1000,
            num_grid_points=10,
            constant_relative_risk_aversion=3.0,
        )

        assert set(
            [
                "optimal_population_percentile",
                "maximum_utility",
                "optimal_transfer_size",
            ]
        ) == set(results_df.columns)
        assert len(results_df) == 2
        assert pytest.approx(
            results_df.optimal_population_percentile.to_numpy(), 1e-2
        ) == [89.0, 89.0]
        assert pytest.approx(results_df.maximum_utility.to_numpy(), 1e-2) == [
            -0.01406,
            -0.00629,
        ]
        assert pytest.approx(results_df.optimal_transfer_size.to_numpy(), 1e-2) == [
            1222.22,
            1222.22,
        ]

    def test_calculate_rank_residuals_table_by_characteristic(self):
        results_df, anova_f_statistic, anova_p_value = (
            calculate_rank_residuals_table_by_characteristic(
                self.household_consumption_data_w_characteristic
            )
        )
        print(results_df)
        print(f"ANOVA F-statistic: {anova_f_statistic}, p-value: {anova_p_value}")
        assert pytest.approx(results_df.loc["group_1", :].to_list(), 1e-2) == [
            0.0,
            0.0247,
        ]
        assert pytest.approx(results_df.loc["group_2", :].to_list(), 1e-2) == [
            0.0101,
            0.0143,
        ]
        assert pytest.approx(anova_f_statistic, 1e-2) == 0.25
        assert pytest.approx(anova_p_value, 1e-2) == 0.6433

    def test_calculate_demographic_parity_table_per_characteristic(self):
        results_df = calculate_demographic_parity_table_per_characteristic(
            self.household_consumption_data_w_characteristic,
            threshold_percentile=50,
        )
        assert set(
            [
                "groundtruth_poverty_percentage",
                "proxy_poverty_percentage",
                "demographic_parity",
                "population_percentage",
            ]
        ) == set(results_df.columns)
        pytest.approx(results_df.loc["group_1", :].to_list(), 1e-2) == [
            50.0,
            50.0,
            1.0,
            0.5,
        ]
        pytest.approx(results_df.loc["group_2", :].to_list(), 1e-2) == [
            50.0,
            50.0,
            1.0,
            0.5,
        ]

    def test_combine_tables_on_characteristic(self):
        combined_table, statistics = combine_tables_on_characteristic(
            self.household_consumption_data_w_characteristic, threshold_percentile=50
        )
        assert set(
            [
                "mean_rank_residual",
                "std_rank_residual",
                "groundtruth_poverty_percentage",
                "proxy_poverty_percentage",
                "demographic_parity",
                "population_percentage",
            ]
        ) == set(combined_table.columns)
        assert set(statistics.columns) == set(
            [
                "anova_f_statistic",
                "anova_p_value",
                "independence_chi2",
                "independence_p_value",
                "precision_chi2",
                "precision_pvalue",
                "recall_chi2",
                "recall_pvalue",
            ]
        )
