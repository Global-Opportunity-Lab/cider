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
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot ROC curves given AUC-ROC with percentile grid data.

    Args:
        auc_roc_precision_recall_with_percentile_grid: DataFrame containing AUC-ROC, precision and recall with percentile grid data with "true_positive_rate", "false_positive_rate", "precision", "recall", and "percentile" columns.
        fixed_groundtruth_percentile (float): Fixed value of percentile for groundtruth consumption values.
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
    return fig, (ax1, ax2)


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
        utility_grid_df: DataFrame containing utility grid data with "percentile" and "utility_proxy" columns.
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
        rank_residuals_series: Series containing rank residuals for each characteristic value; the index holds characteristic values and each element is an array-like of residuals.
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


def plot_all_fairness_metrics_per_characteristic_value(
    all_fairness_metrics_df: pd.DataFrame,
    statistics_df: pd.DataFrame,
    characteristic_name: str = "Characteristic",
    **plot_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot demographic parity per characteristic value.

    Args:
        all_fairness_metrics_df: DataFrame containing data with all fairness metric columns.
        statistics_df: Dataframe containing test statistics for the fairness metrics
        characteristic_name: Name of the characteristic being plotted. Defaults to "Characteristic".
        **plot_kwargs: Additional keyword arguments for matplotlib plot function (e.g., color, linestyle, etc.)
    """
    if not set(
        [
            "independence_1",
            "recall_1",
            "groundtruth_poverty_percentage",
            "proxy_poverty_percentage",
            "demographic_parity",
            "population_percentage",
        ]
    ).issubset(set(all_fairness_metrics_df.columns)):
        raise ValueError(
            "`all_fairness_metrics_df` must contain the following columns: 'independence_1', 'recall_1', 'groundtruth_poverty_percentage', 'proxy_poverty_percentage', 'demographic_parity', 'population_percentage'"
        )

    if not set(
        [
            "independence_chi2",
            "independence_p_value",
            "precision_chi2",
            "precision_pvalue",
            "recall_chi2",
            "recall_pvalue",
        ]
    ).issubset(statistics_df.columns):
        raise ValueError(
            "`statistics_df` must contain the following columns: 'independence_chi2', 'independence_p_value', 'precision_chi2', 'precision_pvalue', 'recall_chi2', 'recall_pvalue"
        )

    with mpl.rc_context(fname=Path(__file__).parent / "../matplotlibrc"):
        fig, axes = plt.subplots(3, 1, figsize=(15, 17))
        fig.subplots_adjust(hspace=0.5)

        # Demographic parity
        axes[0].bar(
            x=np.arange(len(all_fairness_metrics_df)),
            height=all_fairness_metrics_df.groundtruth_poverty_percentage,
            width=0.2,
            label="Groundtruth",
        )
        axes[0].bar(
            x=np.arange(len(all_fairness_metrics_df)) + 0.2,
            height=all_fairness_metrics_df.proxy_poverty_percentage,
            width=0.2,
            label="Proxy",
        )
        axes[0].set_xticks(np.arange(len(all_fairness_metrics_df)) + 0.1)
        axes[0].set_xticklabels(
            [
                ind
                + f"\n Pop. % = {all_fairness_metrics_df.loc[ind, "population_percentage"]: .2f}"
                for ind in all_fairness_metrics_df.index
            ]
        )

        # Annotate bars with demographic parity values
        for i, (_, row) in enumerate(all_fairness_metrics_df.iterrows()):
            axes[0].annotate(
                f"{row.demographic_parity:.3f}",
                xy=(
                    0.1 + i,
                    all_fairness_metrics_df[
                        ["groundtruth_poverty_percentage", "proxy_poverty_percentage"]
                    ]
                    .max()
                    .max()
                    + 1,
                ),
                ha="center",
                fontsize=15,
            )
        axes[0].legend(frameon=True)
        axes[0].set_ylabel("Target percentage")
        axes[0].set_title(
            f"Target percentage and demographic parity per {characteristic_name} Value \n"
            + "(Annotations show demographic parity values) \n"
            + f"Independence between targeted populations (p-value): {statistics_df.independence_p_value[0]:.4f}\n"
        )

        # Recall value
        axes[1].bar(
            x=np.arange(len(all_fairness_metrics_df)),
            height=all_fairness_metrics_df.recall_1 * 100,
            width=0.2,
        )
        axes[1].set_xticks(np.arange(len(all_fairness_metrics_df)) + 0.1)
        axes[1].set_xticklabels(
            [
                ind
                + f"\n Pop. % = {all_fairness_metrics_df.loc[ind, "population_percentage"]: .2f}"
                for ind in all_fairness_metrics_df.index
            ]
        )
        axes[1].set_ylabel("Population percentage")
        axes[1].set_title(
            f"Recall per {characteristic_name} Value\n"
            + f"Recall between targeted populations (p-value): {statistics_df.recall_pvalue[0]:.4f}\n"
        )

        # Precision value
        axes[2].bar(
            x=np.arange(len(all_fairness_metrics_df)),
            height=all_fairness_metrics_df.precision_1 * 100,
            width=0.2,
        )
        axes[2].set_xticks(np.arange(len(all_fairness_metrics_df)) + 0.1)
        axes[2].set_xticklabels(
            [
                ind
                + f"\n Pop. % = {all_fairness_metrics_df.loc[ind, "population_percentage"]: .2f}"
                for ind in all_fairness_metrics_df.index
            ]
        )
        axes[2].set_ylabel("Population percentage")
        axes[2].set_title(
            f"Precision per {characteristic_name} Value\n"
            + f"Precision between targeted populations (p-value): {statistics_df.precision_pvalue[0]:.4f}\n"
        )

        return fig, axes
