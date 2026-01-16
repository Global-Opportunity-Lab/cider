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

from cider.utils import validate_dataframe
from .dependencies import (
    calculate_metrics_binary_valued_consumption,
    where_is_false_positive_rate_nonmonotonic,
    calculate_utility,
    calculate_rank_residuals_by_characteristic,
    calculate_demographic_parity_per_characteristic,
    calculate_independence_btwn_proxy_and_characteristic,
    calculate_precision_and_recall_independence_characteristic,
)
from .schemas import (
    ConsumptionData,
    ConsumptionColumn,
    ConsumptionDataWithCharacteristic,
)
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from typing import Tuple


def compute_auc_roc_precision_recall_with_percentile_grid(
    data: pd.DataFrame,
    fixed_groundtruth_percentile: float,
    num_grid_points: int = 99,
) -> pd.DataFrame:
    """
    Compute AUC-ROC across a grid of percentiles.

    Args:
        data (pd.DataFrame): Data containing 'groundtruth_consumption', 'proxy_consumption', and 'weight' values.
        fixed_groundtruth_percentile (float):  Fixed value of percentile for groundtruth consumption values.
        num_grid_points (int): Number of grid points to compute AUC-ROC.

    Returns:
        pd.DataFrame: DataFrame containing percentiles and corresponding false positive rates, false negative rates, and AUC values.
    """
    # Validate that input data has the required columns
    validate_dataframe(data, required_schema=ConsumptionData)

    # Validate fixed_groundtruth_percentile value
    if not 0.0 < fixed_groundtruth_percentile < 100.0:
        raise ValueError("`fixed_groundtruth_percentile` must be between 0 and 100")

    # Create percentile grid
    percentiles = np.linspace(1, 99, num_grid_points)[::-1]

    results_per_grid = [
        calculate_metrics_binary_valued_consumption(
            data, fixed_groundtruth_percentile, p
        )
        for p in percentiles
    ]
    true_positive_rates = [
        result.true_positive_rate.values[0] for result in results_per_grid
    ]
    false_positive_rates = [
        result.false_positive_rate.values[0] for result in results_per_grid
    ]
    auc_values = [result.auc.values[0] for result in results_per_grid]
    precision_values = [result.precision.values[0] for result in results_per_grid]
    recall_values = [result.recall.values[0] for result in results_per_grid]

    nonmonotonic_indices = where_is_false_positive_rate_nonmonotonic(
        np.array(false_positive_rates)
    )

    if len(nonmonotonic_indices) > 0:
        false_positive_rates = [
            false_positive_rates[i]
            for i in range(len(false_positive_rates))
            if i not in nonmonotonic_indices
        ]
        true_positive_rates = [
            true_positive_rates[i]
            for i in range(len(true_positive_rates))
            if i not in nonmonotonic_indices
        ]
        percentiles = [
            percentiles[i]
            for i in range(len(percentiles))
            if i not in nonmonotonic_indices
        ]
        auc_values = [
            auc_values[i]
            for i in range(len(auc_values))
            if i not in nonmonotonic_indices
        ]
        precision_values = [
            precision_values[i]
            for i in range(len(precision_values))
            if i not in nonmonotonic_indices
        ]
        recall_values = [
            recall_values[i]
            for i in range(len(recall_values))
            if i not in nonmonotonic_indices
        ]

    results_df = pd.DataFrame(
        {
            "percentile": percentiles,
            "precision": precision_values,
            "recall": recall_values,
            "true_positive_rate": true_positive_rates,
            "false_positive_rate": false_positive_rates,
            "auc": auc_values,
        }
    )
    return results_df


def compute_utility_grid(
    data: pd.DataFrame,
    cash_transfer_amount: float,
    num_grid_points: int = 100,
    constant_relative_risk_aversion: float = 3.0,
) -> pd.DataFrame:
    """
    Compute utility across a grid of percentiles.

    Args:
        data (pd.DataFrame): Data containing consumption values and weights.
        cash_transfer_amount (float): Amount of cash transfer if the entire population was to receive a cash transfer.
        num_grid_points (int): Number of grid points to compute utility.
        constant_relative_risk_aversion (float): Coefficient of relative risk aversion.

    Returns:
        pd.DataFrame: DataFrame containing percentiles and corresponding utility values.
    """
    # Validate that input data has the required columns
    validate_dataframe(data, required_schema=ConsumptionData)

    # Create percentile grid
    percentiles = np.linspace(1, 100, num_grid_points)
    utilities: dict[str, list[float]] = {
        col: [] for col in [ConsumptionColumn.GROUNDTRUTH, ConsumptionColumn.PROXY]
    }
    cash_transfer_amounts: dict[str, list[float]] = {
        col: [] for col in [ConsumptionColumn.GROUNDTRUTH, ConsumptionColumn.PROXY]
    }

    total_cash_available = cash_transfer_amount * data["weight"].sum()

    for percentile in percentiles:
        # Compute utility per consumption column
        for consumption_col in [ConsumptionColumn.GROUNDTRUTH, ConsumptionColumn.PROXY]:

            is_cash_transferred = data[consumption_col] <= np.percentile(
                data[consumption_col], percentile
            )
            cash_transfer_amount = (
                total_cash_available / (is_cash_transferred * data.weight).sum()
            )
            utility = calculate_utility(
                data,
                threshold_percentile=percentile,
                consumption_column=consumption_col,
                cash_transfer_amount=cash_transfer_amount,
                constant_relative_risk_aversion=constant_relative_risk_aversion,
            )

            cash_transfer_amounts[consumption_col].append(cash_transfer_amount)
            utilities[consumption_col].append(utility)

    results_df = pd.DataFrame(
        {
            "percentile": percentiles,
            "cash_transfer_amount_groundtruth": cash_transfer_amounts[
                ConsumptionColumn.GROUNDTRUTH
            ],
            "cash_transfer_amount_proxy": cash_transfer_amounts[
                ConsumptionColumn.PROXY
            ],
            "utility_groundtruth": utilities[ConsumptionColumn.GROUNDTRUTH],
            "utility_proxy": utilities[ConsumptionColumn.PROXY],
        }
    )
    return results_df


def calculate_optimal_utility_and_cash_transfer_size_table(
    data: pd.DataFrame,
    cash_transfer_amount: float,
    num_grid_points: int = 100,
    constant_relative_risk_aversion: float = 3.0,
) -> pd.DataFrame:
    """
    Calculate optimal utility and cash transfer size for groundtruth and proxy consumption.

    Args:
        data (pd.DataFrame): Data containing consumption values and weights.
        cash_transfer_amount (float): Amount of cash transfer if the entire population was to receive a cash transfer.
        num_grid_points (int): Number of grid points to compute utility.
        constant_relative_risk_aversion (float): Coefficient of relative risk aversion.

    Returns:
        pd.DataFrame: DataFrame containing optimal cash transfer sizes and utilities for groundtruth and proxy consumption.
    """
    # Validate that input data has the required columns
    validate_dataframe(data, required_schema=ConsumptionData)

    # Compute utility grid
    utility_grid_df = compute_utility_grid(
        data,
        cash_transfer_amount=cash_transfer_amount,
        num_grid_points=num_grid_points,
        constant_relative_risk_aversion=constant_relative_risk_aversion,
    )

    # Find optimal utility and corresponding cash transfer size
    optimal_groundtruth_idx = utility_grid_df["utility_groundtruth"].idxmax()
    optimal_proxy_idx = utility_grid_df["utility_proxy"].idxmax()

    results = pd.DataFrame(
        {
            "optimal_population_percentile": [
                utility_grid_df.loc[optimal_groundtruth_idx, "percentile"],
                utility_grid_df.loc[optimal_proxy_idx, "percentile"],
            ],
            "maximum_utility": [
                utility_grid_df.loc[optimal_groundtruth_idx, "utility_groundtruth"],
                utility_grid_df.loc[optimal_proxy_idx, "utility_proxy"],
            ],
            "optimal_transfer_size": [
                utility_grid_df.loc[
                    optimal_groundtruth_idx, "cash_transfer_amount_groundtruth"
                ],
                utility_grid_df.loc[optimal_proxy_idx, "cash_transfer_amount_proxy"],
            ],
        },
        index=[col.value for col in ConsumptionColumn],
    )

    return results


def calculate_rank_residuals_table_by_characteristic(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Calculate rank residuals by characteristic.

    Args:
        data (pd.DataFrame): Data containing consumption values, weights, and characteristic.

    Returns:
        pd.DataFrame: DataFrame containing rank residuals statistics by characteristic.
    """
    # Validate that input data has the required columns
    validate_dataframe(data, required_schema=ConsumptionDataWithCharacteristic)

    results_df = calculate_rank_residuals_by_characteristic(data)
    means = [np.mean(r) for r in results_df]
    stds = [np.std(r) for r in results_df]
    anova_results = f_oneway(*tuple(results_df))

    return (
        pd.DataFrame(
            {
                "mean_rank_residual": means,
                "std_rank_residual": stds,
            },
            index=results_df.index,
        ),
        anova_results.statistic,
        anova_results.pvalue,
    )


def calculate_demographic_parity_table_per_characteristic(
    data: pd.DataFrame,
    threshold_percentile: float,
) -> pd.DataFrame:
    """
    Calculate demographic parity per characteristic.

    Args:
        data (pd.DataFrame): Data containing consumption values, weights, and characteristic.
        threshold_percentile (float): Percentile threshold for targeting.

    Returns:
        pd.DataFrame: DataFrame containing demographic parity statistics by characteristic.
    """
    # Validate that input data has the required columns
    validate_dataframe(data, required_schema=ConsumptionDataWithCharacteristic)

    results = calculate_demographic_parity_per_characteristic(
        data, threshold_percentile
    )
    results["population_percentage"] = data.groupby("characteristic").apply(
        lambda x: 100 * x["weight"].sum() / data["weight"].sum(), include_groups=False
    )

    return results


def combine_tables_on_characteristic(
    data: pd.DataFrame,
    threshold_percentile: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine rank residuals table, demographic parity table, and independence, precision and recall values on characteristic.

    Args:
        data (pd.DataFrame): Data containing consumption values, weights, and characteristic.
        threshold_percentile (float): Percentile threshold for targeting.

    Returns:
        pd.DataFrame: Combined DataFrame.
    """
    # Validate that input data has the required columns
    validate_dataframe(data, required_schema=ConsumptionDataWithCharacteristic)

    # Calculate individual components
    rank_residuals_table, anova_statistic, anova_pvalue = (
        calculate_rank_residuals_table_by_characteristic(data)
    )
    demographic_parity_table = calculate_demographic_parity_table_per_characteristic(
        data, threshold_percentile
    )

    pivot_independence, independence = (
        calculate_independence_btwn_proxy_and_characteristic(data, threshold_percentile)
    )
    independence_chi2, independence_p_value = independence.loc[0, :].to_list()

    pivot_precision, pivot_recall, precision_recall = (
        calculate_precision_and_recall_independence_characteristic(
            data, threshold_percentile, threshold_percentile
        )
    )
    precision_chi2, precision_pvalue = precision_recall.loc["precision", :].to_list()
    recall_chi2, recall_pvalue = precision_recall.loc["recall", :].to_list()

    # Combine tables
    pivot_independence.columns = [
        f"independence_{col}" for col in pivot_independence.columns
    ]
    pivot_precision.columns = [f"precision_{col}" for col in pivot_precision.columns]
    pivot_recall.columns = [f"recall_{col}" for col in pivot_recall.columns]
    combined_pivot = pd.concat(
        [pivot_independence, pivot_precision, pivot_recall], axis=1
    )

    combined_table = rank_residuals_table.merge(
        demographic_parity_table,
        left_index=True,
        right_index=True,
        how="inner",
    )

    all_fairness_metrics_df = combined_pivot.merge(
        combined_table, left_index=True, right_index=True
    )

    # Get statistics
    statistics = pd.DataFrame(
        {
            "anova_f_statistic": [anova_statistic],
            "anova_p_value": [anova_pvalue],
            "independence_chi2": [independence_chi2],
            "independence_p_value": [independence_p_value],
            "precision_chi2": [precision_chi2],
            "precision_pvalue": [precision_pvalue],
            "recall_chi2": [recall_chi2],
            "recall_pvalue": [recall_pvalue],
        }
    )

    return all_fairness_metrics_df, statistics
