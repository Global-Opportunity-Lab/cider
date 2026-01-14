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
):
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
