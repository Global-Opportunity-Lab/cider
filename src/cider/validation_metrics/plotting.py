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
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def plot_roc_precision_recall_curves(
    auc_roc_precision_recall_with_percentile_grid: pd.DataFrame,
    fixed_groundtruth_percentile: float,
    **plotting_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot ROC curves given AUC-ROC with percentile grid data.

    Args:
        auc_roc_with_percentile_grid: DataFrame containing AUC-ROC, precision and recall with percentile grid data with "true_positive_rate", "false_positive_rate", "precision", "recall", and "percentile" columns.
        fixed_groundtruth_percentile (float): fixed_groundtruth_percentile (float):  Fixed value of percentile for groundtruth consumption values.
        **plotting_kwargs: Additional keyword arguments for matplotlib plot function (e.g., color, linestyle, etc.)
    """

    if not set(
        [
            "true_positive_rate",
            "false_positive_rate",
            "precision",
            "recall",
            "percentile",
        ]
    ).issubset(auc_roc_precision_recall_with_percentile_grid.columns):
        raise ValueError(
            "DataFrame must contain 'true_positive_rate', 'false_positive_rate', and 'percentile' columns"
        )
    if not 0 < fixed_groundtruth_percentile < 100:
        raise ValueError("`fixed_groundtruth_value` must be between 0. and 100.")

    with mpl.rc_context(fname=Path(__file__).parent / "../matplotlibrc"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
        ax1.plot(
            auc_roc_precision_recall_with_percentile_grid["false_positive_rate"] * 100,
            auc_roc_precision_recall_with_percentile_grid["true_positive_rate"] * 100,
            **plotting_kwargs,
        )
        ax1.plot(
            [0, 100], [0, 100], linestyle="--", color="gray", label="x=y"
        )  # Diagonal line
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title(
            f"ROC Curve: groundtruth percentile = {fixed_groundtruth_percentile}%"
        )
        ax1.legend()

        ax2.plot(
            auc_roc_precision_recall_with_percentile_grid["percentile"],
            auc_roc_precision_recall_with_percentile_grid["precision"] * 100,
            **plotting_kwargs,
            label="Precision",
        )
        ax2.plot(
            auc_roc_precision_recall_with_percentile_grid["percentile"],
            auc_roc_precision_recall_with_percentile_grid["recall"] * 100,
            **plotting_kwargs,
            label="Recall",
        )
        ax2.set_xlabel("Percentile: share of population targeted")
        ax2.set_ylabel("Precision / Recall")
        ax2.set_title(
            f"Precision-Recall Curve: groundtruth percentile = {fixed_groundtruth_percentile}%"
        )
        ax2.legend()

        fig.tight_layout()
    return fig, ax1


def plot_utility_values(
    utility_grid_df: pd.DataFrame,
    optimal_percentile: float,
    optimal_utility: int,
    cash_transfer_amount: float,
    constant_relative_risk_aversion: float,
    **plotting_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot utility values given utility grid data.

    Args:
        utility_grid_df: DataFrame containing utility grid data with "percentile" and "utilit_proxy" columns.
        optimal_percentile: Optimal percentile value.
        optimal_utility: Optimal utility value.
        cash_transfer_amount: Cash transfer amount used in utility calculation.
        constant_relative_risk_aversion: Constant relative risk aversion parameter used in utility calculation.
        **plotting_kwargs: Additional keyword arguments for matplotlib plot function (e.g., color, linestyle, etc.)
    """

    if not set(["percentile", "utility_proxy"]).issubset(utility_grid_df.columns):
        raise ValueError(
            "DataFrame must contain 'percentile' and 'utility_proxy' columns"
        )

    with mpl.rc_context(fname=Path(__file__).parent / "../matplotlibrc"):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(
            utility_grid_df["percentile"],
            utility_grid_df["utility_proxy"],
            **plotting_kwargs,
        )
        ax.scatter(optimal_percentile, optimal_utility, color=["k"], s=50)
        ax.set_xlabel("Percentile: share of population targeted")
        ax.set_ylabel("Utility with consumption proxy")
        ax.set_title(
            f"Utility Values: \n Cash transfer amount = {round(cash_transfer_amount, 2)}, CRRA = {constant_relative_risk_aversion}"
        )

        fig.tight_layout()
    return fig, ax


def plot_rank_residual_distributions_per_characteristic_value(
    rank_residuals_series: pd.Series,
    characteristic_name: str = "Characteristic",
    **plot_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot rank residual distributions per characteristic value.

    Args:
        rank_residuals_df: DataFrame containing rank residuals with "rank_residual" and "characteristic_value" columns.
        characteristic_name: Name of the characteristic being plotted. Defaults to "Characteristic".
        **plot_kwargs: Additional keyword arguments for matplotlib plot function (e.g., color, linestyle, etc.)
    """
    normalized_residuals = rank_residuals_series.apply(
        lambda x: np.array(x) / np.max(x)
    )

    with mpl.rc_context(fname=Path(__file__).parent / "../matplotlibrc"):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.axvline(x=0, color="grey", linestyle="--")
        ax.boxplot(normalized_residuals, orientation="horizontal", **plot_kwargs)
        ax.set_yticklabels(normalized_residuals.index)
        ax.set_ylabel(f"{characteristic_name} Value")
        ax.set_xlabel("Normalized Rank Residuals")
        ax.set_title(f"Rank Residual Distributions per {characteristic_name} Value")
        fig.tight_layout()

        return fig, ax


def plot_demographic_parity_per_characteristic_value(
    demographic_parity_df: pd.DataFrame,
    characteristic_name: str = "Characteristic",
    **plot_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot demographic parity per characteristic value.

    Args:
        demographic_parity_df: DataFrame containing demographic parity data with "demographic_parity", "groundtruth_poverty_percentage", "proxy_poverty_percentage", and "population_percentage" columns.
        characteristic_name: Name of the characteristic being plotted. Defaults to "Characteristic".
        **plot_kwargs: Additional keyword arguments for matplotlib plot function (e.g., color, linestyle, etc.)
    """
    with mpl.rc_context(fname=Path(__file__).parent / "../matplotlibrc"):
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.bar(
            x=np.arange(len(demographic_parity_df)),
            height=demographic_parity_df.population_percentage,
            width=0.2,
            label="Population",
        )
        ax.bar(
            x=np.arange(len(demographic_parity_df)) + 0.2,
            height=demographic_parity_df.groundtruth_poverty_percentage,
            width=0.2,
            label="Groundtruth",
        )
        ax.bar(
            x=np.arange(len(demographic_parity_df)) + 0.4,
            height=demographic_parity_df.proxy_poverty_percentage,
            width=0.2,
            label="Proxy",
        )
        ax.set_xticks(np.arange(len(demographic_parity_df)) + 0.2)
        ax.set_xticklabels(demographic_parity_df.index)

        # Annotate bars with demographic parity values
        for i, (_, row) in enumerate(demographic_parity_df.iterrows()):
            ax.annotate(
                f"{row.demographic_parity:.3f}",
                xy=(
                    0.2 + i,
                    demographic_parity_df.drop(columns=["demographic_parity"])
                    .max()
                    .max()
                    + 1,
                ),
                ha="center",
                fontsize=15,
            )
        ax.legend(fontsize=10, frameon=True)
        ax.set_xlabel(f"{characteristic_name} Value")
        ax.set_ylabel("Population / target percentage")

        fig.suptitle(
            f"Population percentage and demographic parity per {characteristic_name} Value \n (Annotations show Demographic Parity Values)"
        )

        return fig, ax
