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


from cider.utils import _validate_dataframe
from .schemas import (
    ConsumptionData,
    ConsumptionColumn,
    ConsumptionDataWithCharacteristic,
)
from scipy.stats import rankdata, chi2_contingency
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def convert_threshold_to_percentile(
    threshold: float | list[float],
    data: pd.DataFrame,
    consumption_column: ConsumptionColumn = ConsumptionColumn.GROUNDTRUTH,
) -> float:
    """Convert a threshold value to its corresponding percentile in the data.

    Args:
        threshold (float | list[float]): The threshold value(s) to convert.
        data (pd.DataFrame): Data containing 'groundtruth_consumption', 'proxy_consumption', and 'weight' values.
        consumption_column (ConsumptionColumn): The column name for consumption values in the data.

    Returns:
        float: The percentile corresponding to the threshold value.
    """
    # Validate that input data has the required columns
    _validate_dataframe(data, required_schema=ConsumptionData)

    # Calculate and return the percentile
    consumption_data = data[consumption_column.value].to_numpy()
    weights = data["weight"].to_numpy()
    sorted_indices = np.argsort(consumption_data)
    sorted_consumption = consumption_data[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute the cumulative weights
    cumulative_weights = np.cumsum(sorted_weights)
    normalized_cumulative_weights = (cumulative_weights / cumulative_weights[-1]) * 100

    # Convert threshold to percentile
    percentile = np.atleast_1d(threshold)
    percentile = np.interp(
        percentile, sorted_consumption, normalized_cumulative_weights
    )

    return percentile if len(percentile) > 1 else percentile[0]


def calculate_weighted_spearmanr(
    data: pd.DataFrame, significant_digits: int = 2
) -> float:
    """
    Calculate the weighted Spearman correlation
    Args:
        data (pd.DataFrame): DataFrame containing 'groundtruth_consumption', 'proxy_consumption', and 'weight' columns.
    Returns:
        float: Weighted Spearman correlation coefficient.
    """
    # Validate that input data has the required columns
    _validate_dataframe(data, required_schema=ConsumptionData)

    # Rank the groundtruth and proxy consumption values
    rank_groundtruth = rankdata(
        data["groundtruth_consumption"], method="ordinal"
    )  # use 'ordinal' to avoid ties
    rank_proxy = rankdata(
        data["proxy_consumption"], method="ordinal"
    )  # use 'ordinal' to avoid ties

    # Compute weighted Spearman correlation
    covariance = np.cov(
        rank_groundtruth,
        rank_proxy,
        aweights=data["weight"],
    )
    std_groundtruth = np.sqrt(covariance[0, 0])
    std_proxy = np.sqrt(covariance[1, 1])
    spearmans_r = covariance[0, 1] / (std_groundtruth * std_proxy)

    return round(spearmans_r, significant_digits)


def calculate_weighted_pearsonr(
    data: pd.DataFrame, significant_digits: int = 2
) -> float:
    """
    Calculate the weighted Pearson correlation
    Args:
        data (pd.DataFrame): DataFrame containing 'groundtruth_consumption', 'proxy_consumption', and 'weight' columns.
    Returns:
        float: Weighted Pearson correlation coefficient.
    """
    # Validate that input data has the required columns
    _validate_dataframe(data, required_schema=ConsumptionData)

    # Compute weighted Pearson correlation
    covariance_matrix = np.cov(
        data["groundtruth_consumption"],
        data["proxy_consumption"],
        aweights=data["weight"],
    )

    pearsons_r = covariance_matrix[0, 1] / np.sqrt(
        covariance_matrix[0, 0] * covariance_matrix[1, 1]
    )
    return round(pearsons_r, significant_digits)


def calculate_metrics_binary_valued_consumption(
    data: pd.DataFrame,
    groundtruth_threshold_percentile: float,
    proxy_threshold_percentile: float,
) -> pd.DataFrame:
    """
    Calculate AUC curves for binary-valued (below / above threshold) consumption data at specified thresholds.

    Args:
        data (pd.DataFrame): DataFrame containing 'groundtruth_consumption', 'proxy_consumption' and 'weight' columns.
        groundtruth_threshold_percentile (float): Percentile threshold to use for groundtruth consumption calculation.
        proxy_threshold_percentile (float): Percentile threshold to use for proxy consumption calculation.
    Returns:
        pd.DataFrame: DataFrame containing binary metrics.
    """
    # Validate that input data has the required columns
    _validate_dataframe(data, required_schema=ConsumptionData)

    # Validate threshold values are correct
    if not (0.0 < groundtruth_threshold_percentile < 100):
        raise ValueError("groundtruth_threshold_percentile must be between 0 and 100")
    if not (0.0 < proxy_threshold_percentile < 100):
        raise ValueError("proxy_threshold_percentile must be between 0 and 100")

    # Binarize consumption values based on thresholds
    groundtruth_threshold_value = np.percentile(
        data["groundtruth_consumption"], groundtruth_threshold_percentile
    )
    proxy_threshold_value = np.percentile(
        data["proxy_consumption"], proxy_threshold_percentile
    )

    groundtruth_binary = (
        data["groundtruth_consumption"] <= groundtruth_threshold_value
    ).astype(int)
    proxy_binary = (data["proxy_consumption"] <= proxy_threshold_value).astype(int)

    # Calculate metrics
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(
        groundtruth_binary, proxy_binary, sample_weight=data["weight"]
    ).ravel()

    # Calculate rates
    results = {
        "accuracy": (true_pos + true_neg)
        / (true_pos + true_neg + false_pos + false_neg),
        "precision": true_pos / (true_pos + false_pos),
        "recall": true_pos / (true_pos + false_neg),
        "true_positive_rate": true_pos / (true_pos + false_neg),
        "false_positive_rate": false_pos / (false_pos + true_neg),
        "auc": roc_auc_score(
            groundtruth_binary, -data["proxy_consumption"], sample_weight=data["weight"]
        ),
        "roc_curve": roc_curve(
            groundtruth_binary, -data["proxy_consumption"], sample_weight=data["weight"]
        ),
        "spearman_r": calculate_weighted_spearmanr(data),
        "pearson_r": calculate_weighted_pearsonr(data),
    }

    return pd.DataFrame([results])


def calculate_utility(
    data: pd.DataFrame,
    threshold_percentile: float,
    consumption_column: ConsumptionColumn,
    cash_transfer_amount: float,
    constant_relative_risk_aversion: float = 3.0,
) -> float:
    """
    Computes the constant relative risk-aversion (CRRA) [Hanna & Olken (2018)] utility when P% of the population is
    targeted and they receive transfers of size 'transfer_size'

    Args:
        data (pd.DataFrame): DataFrame containing 'groundtruth_consumption', 'proxy_consumption' and 'weight'
        threshold_percentile (float): Percentile threshold to use for consumption calculation.
        consumption_column (ConsumptionColumn): Enum indicating which consumption column to use.
        cash_transfer_amount (float): Amount of cash transfer given to targeted households.
        constant_relative_risk_aversion (float): Coefficient of relative risk aversion (CRRA) utility function.

    Returns:
        float: The utility obtained by targeting the specified percentile of the population with the specified cash transfer amount.
    """

    # Validate that input data has the required columns
    _validate_dataframe(data, required_schema=ConsumptionData)

    # Validate threshold values are correct
    if not (0.0 <= threshold_percentile <= 100):
        raise ValueError("threshold_percentile must be between 0 and 100")

    # Compute utility
    threshold_value = np.percentile(
        data[consumption_column.value], threshold_percentile
    )
    is_cash_transferred = (data[consumption_column.value] < threshold_value).astype(
        float
    )
    benefits = is_cash_transferred * cash_transfer_amount
    utility = (
        (data[consumption_column.value] + benefits * data.weight / data.weight.sum())
        ** (1 - constant_relative_risk_aversion)
    ) / (1 - constant_relative_risk_aversion)
    return utility.sum()


def where_is_false_positive_rate_nonmonotonic(
    false_positive_rates: np.ndarray,
) -> np.ndarray:
    """
    Check if false positive rates are strictly increasing.

    Args:
        false_positive_rates (np.ndarray): Array of false positive rates.
    Returns:
        bool: True if false positive rates are strictly increasing, False otherwise.
    """
    return np.argwhere(false_positive_rates[1:] > false_positive_rates[:-1])


def calculate_rank_residuals_by_characteristic(data: pd.DataFrame):
    """
    Calculate rank residuals between groundtruth and proxy consumption by characteristic, to check for consistent biases in ranking.

    Args:
        data (pd.DataFrame): DataFrame containing 'groundtruth_consumption', 'proxy_consumption', 'weight', and 'characteristic' columns.
        unique_characteristic_values (set): Set of allowed values for the characteristic column.
    Returns:
        pd.Series: Series containing rank residuals grouped by characteristic.
    """
    # Validate that input data has the required columns
    _validate_dataframe(data, required_schema=ConsumptionDataWithCharacteristic)

    # Compute rank residuals by characteristic
    data_copy = data.copy()
    groundtruth_ranks = rankdata(
        data["groundtruth_consumption"], method="ordinal"
    )  # use 'ordinal' to avoid ties
    proxy_ranks = rankdata(
        data["proxy_consumption"], method="ordinal"
    )  # use 'ordinal' to avoid ties
    data_copy["rank_residual"] = (
        data.weight
        * (groundtruth_ranks - proxy_ranks)
        / (data.weight.sum() * proxy_ranks.max())
    )

    return data_copy.groupby("characteristic")["rank_residual"].apply(list)


def calculate_demographic_parity_per_characteristic(
    data: pd.DataFrame, threshold_percentile: float
) -> pd.DataFrame:
    """
    Calculate demographic parity difference in targeting rates per characteristic groups.
    Demographic parity is defined as the difference in the population targeted using the groundtruth consumption and the proxy variable.

    Args:
        data (pd.DataFrame): DataFrame containing 'groundtruth_consumption', 'proxy_consumption', 'weight', and 'characteristic' columns.
        threshold_percentile (float): Percentile threshold to use for consumption calculation.

    Returns:
        pd.DataFrame: DataFrame containing demographic parity differences per characteristic.
    """
    # Validate that input data has the required columns
    _validate_dataframe(data, required_schema=ConsumptionDataWithCharacteristic)

    # Validate threshold values are correct
    if not (0.0 < threshold_percentile < 100):
        raise ValueError("threshold_percentile must be between 0 and 100")

    # Calculate demographic parity per characteristic
    proxy_threshold_value = np.percentile(
        data["proxy_consumption"], threshold_percentile
    )
    groundtruth_threshold_value = np.percentile(
        data["groundtruth_consumption"], threshold_percentile
    )

    data_copy = data.copy()
    data_copy["is_targeted_proxy"] = (
        data_copy["proxy_consumption"] <= proxy_threshold_value
    ).astype(int)
    data_copy["is_targeted_groundtruth"] = (
        data_copy["groundtruth_consumption"] <= groundtruth_threshold_value
    ).astype(int)
    total_population = data_copy["weight"].sum()

    data_grouped = data_copy.groupby("characteristic").apply(
        lambda x: pd.Series(
            {
                "groundtruth_poverty_percentage": 100
                * (x.is_targeted_groundtruth * x.weight).sum()
                / total_population,
                "proxy_poverty_percentage": 100
                * (x.is_targeted_proxy * x.weight).sum()
                / total_population,
            },
        ),
        include_groups=False,
    )
    data_grouped["demographic_parity"] = (
        data_grouped["proxy_poverty_percentage"]
        - data_grouped["groundtruth_poverty_percentage"]
    )

    return data_grouped


def calculate_independence_btwn_proxy_and_characteristic(
    data: pd.DataFrame, threshold_percentile: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate independence between proxy variable and characteristic at a specified threshold.
    Independence is defined as the difference in targeting rates across characteristic groups using the proxy variable.

    Args:
        data (pd.DataFrame): DataFrame containing 'proxy_consumption', 'weight', and 'characteristic' columns.
        threshold_percentile (float): Percentile threshold to use for consumption calculation.
    Returns:
        chi2 statistic and p-value from chi-squared test for independence.
    """
    # Validate that input data has the required columns
    _validate_dataframe(data, required_schema=ConsumptionDataWithCharacteristic)

    # Validate threshold values are correct
    if not (0.0 < threshold_percentile < 100):
        raise ValueError("threshold_percentile must be between 0 and 100")

    # Calculate independence between proxy and characteristic
    proxy_threshold_value = np.percentile(
        data["proxy_consumption"], threshold_percentile
    )

    data_copy = data.copy()
    data_copy["is_targeted_proxy"] = (
        data_copy["proxy_consumption"] <= proxy_threshold_value
    ).astype(int)

    pivot = data_copy.pivot_table(
        index="characteristic",
        columns="is_targeted_proxy",
        values="weight",
        aggfunc="sum",
        fill_value=0,
    )
    chi2, p_value, _, _ = chi2_contingency(pivot)

    return pivot, pd.DataFrame({"chi2_statistic": [chi2], "p_value": [p_value]})


def calculate_precision_and_recall_independence_characteristic(
    data: pd.DataFrame,
    groundtruth_threshold_percentile: float,
    proxy_threshold_percentile: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate precision and recall independence per characteristic.

    Args:
        data (pd.DataFrame): DataFrame containing 'groundtruth_consumption', 'proxy_consumption', 'weight', and 'characteristic' columns.
        groundtruth_threshold_percentile (float): Percentile threshold to use for groundtruth consumption calculation.
        proxy_threshold_percentile (float): Percentile threshold to use for proxy consumption calculation.
    Returns:
        tuple[float, float, float]: Chi-squared statistics and p-values for precision and recall independence per characteristic.
    """
    # Validate that input data has the required columns
    _validate_dataframe(data, required_schema=ConsumptionDataWithCharacteristic)

    # Validate threshold values are correct
    if not (0.0 < groundtruth_threshold_percentile < 100):
        raise ValueError("groundtruth_threshold_percentile must be between 0 and 100")
    if not (0.0 < proxy_threshold_percentile < 100):
        raise ValueError("proxy_threshold_percentile must be between 0 and 100")

    # Binarize consumption values based on thresholds
    groundtruth_threshold_value = np.percentile(
        data["groundtruth_consumption"], groundtruth_threshold_percentile
    )
    proxy_threshold_value = np.percentile(
        data["proxy_consumption"], proxy_threshold_percentile
    )

    data_copy = data.copy()
    data_copy["groundtruth_binary"] = (
        data_copy["groundtruth_consumption"] <= groundtruth_threshold_value
    ).astype(int)
    data_copy["proxy_binary"] = (
        data_copy["proxy_consumption"] <= proxy_threshold_value
    ).astype(int)

    filtered_precision_data = data_copy.loc[data_copy.proxy_binary == 1]
    filtered_recall_data = data_copy.loc[data_copy.groundtruth_binary == 1]

    pivot_precision = filtered_precision_data.pivot_table(
        index="characteristic",
        columns="groundtruth_binary",
        values="weight",
        aggfunc="sum",
        fill_value=0,
    )
    pivot_recall = filtered_recall_data.pivot_table(
        index="characteristic",
        columns="proxy_binary",
        values="weight",
        aggfunc="sum",
        fill_value=0,
    )

    chi2_precision, p_value_precision, _, _ = chi2_contingency(pivot_precision)
    chi2_recall, p_value_recall, _, _ = chi2_contingency(pivot_recall)

    return (
        pivot_precision,
        pivot_recall,
        pd.DataFrame(
            {
                "chi2_statistic": [chi2_precision, chi2_recall],
                "p_value": [p_value_precision, p_value_recall],
            },
            index=["precision", "recall"],
        ),
    )
