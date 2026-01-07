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

# TODO: parallelize lasso and forward selection
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import]
from deprecated.helpers.ml_utils import confusion_matrix, strictly_increasing
from deprecated.helpers.plot_utils import clean_plot, mtick
from deprecated.helpers.utils import make_dir
from pandas import DataFrame as PandasDataFrame
from scipy.stats import spearmanr  # type: ignore[import]
from sklearn.metrics import auc, roc_auc_score, roc_curve  # type: ignore[import]

from .datastore import DataStore, DataType


class Targeting:

    def __init__(self, datastore: DataStore, clean_folders: bool = False) -> None:
        self.cfg = datastore.cfg
        self.ds = datastore
        self.outputs = self.cfg.path.working.directory_path / "targeting"

        # Prepare working directories, set color palette
        make_dir(self.outputs, clean_folders)
        self.default_colors = sns.color_palette("Set2", 100)

        # Load data into datastore
        data_type_map = {DataType.TARGETING: None}
        self.ds.load_data(data_type_map=data_type_map)

    @staticmethod
    def threshold_to_percentile(
        p: Optional[Union[float, int]],
        t: Optional[Union[float, int]],
        data: PandasDataFrame,
        var: str,
        test_percentiles: bool = True,
    ) -> Union[float, int]:
        """
        If a percentile is provided, checks if it's between 0 and 100 and if so returns it; if a threshold is provided,
        returns the fraction of records in the dataset and for column 'var' that are below the threshold.

        Args:
            p: The percentile.
            t: The threshold.
            data: The pandas dataframe containing the data.
            var: The column with the records to use to compute the percentile.
            test_percentiles: If true checks that the percentile provided is correct.

        Returns: The percentile corresponding to the input (threshold or percentile) provided.
        """

        # Make sure at least one of t or p is provided
        if p is None and t is None:
            raise ValueError("Must provide percentage targeting or threshold targeting")

        # Make sure not both t and p are provided
        if p is not None and t is not None:
            raise ValueError(
                "Cannot provide both percentage and threshold targeting for var1"
            )

        # Make sure percentile is within range (0 to 100 exclusive)
        if test_percentiles and (p == 0 or p == 100):
            raise ValueError(
                "Percentage targeting must be between 0 and 100 (exclusive)."
            )

        # If t is provided, convert threshold to percentile
        if t is not None:
            p = 100 * (len(data[data[var] < t]) / len(data))

        return p

    def pearson(self, var1: str, var2: str, weighted: bool = False) -> float:
        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting
        return np.corrcoef(data[var1].astype("float"), data[var2].astype("float"))[0][1]

    def spearman(self, var1: str, var2: str, weighted: bool = False) -> float:
        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting
        return spearmanr(data[var1].astype("float"), data[var2].astype("float"))[0]

    def binary_metrics(
        self,
        var1: str,
        var2: str,
        p1: Union[int, float],
        p2: Optional[Union[int, float]],
        t1: Optional[Union[int, float]] = None,
        t2: Optional[Union[int, float]] = None,
        weighted: bool = False,
    ) -> Dict[str, float]:
        """
        Computes several classification metrics for two numerical variables, usually a ground-truth one and its
        prediction; these are first converted into binary variables using the percentiles or thresholds provided.

        Args:
            var1: The first variable to use in the analysis; it has to be a column in the targeting dataset.
            var2: The second variable to use in the analysis; it has to be a column in the targeting dataset.
            p1: Percentile for the first variable.
            p2: Percentile for the second variable.
            t1: Threshold for the first variable.
            t2: Threshold for the second variable.
            weighted: If True the weighted version of the dataset will be used.

        Returns: A dictionary with results on accuracy, precision, recall, true and false positive rates.

        """
        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting
        results = {}

        # If thresholds are provided, convert to percentiles
        p1 = self.threshold_to_percentile(p1, t1, data, var1)
        p2 = self.threshold_to_percentile(p2, t2, data, var2)

        # Stack var1 and var2 together
        a = np.vstack([data[var1].values.flatten(), data[var2].values.flatten()])

        # Order by var1, assign targeting
        num_ones = int((p1 / 100) * len(data))
        num_zeros = len(data) - num_ones
        targeting_vector = np.concatenate([np.ones(num_ones), np.zeros(num_zeros)])
        a = a[:, a[0, :].argsort()]
        a[0, :] = targeting_vector

        # Reshuffle
        np.random.seed(1)
        a = a[:, np.random.rand(a.shape[1]).argsort()]

        # Order by var2, assign targeting
        num_ones = int((p2 / 100) * len(data))
        num_zeros = len(data) - num_ones
        targeting_vector = np.concatenate([np.ones(num_ones), np.zeros(num_zeros)])
        a = a[:, a[1, :].argsort()]
        a[1, :] = targeting_vector

        # Calculate confusion matrix and from there the binary metrics
        tn, fp, fn, tp = confusion_matrix(a[0, :], a[1, :]).ravel()
        results["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
        results["precision"] = tp / (tp + fp)
        results["recall"] = tp / (tp + fn)
        results["tpr"] = tp / (tp + fn)
        results["fpr"] = fp / (fp + tn)
        return results

    def auc_threshold(
        self,
        var1: str,
        var2: str,
        p: Optional[Union[int, float]],
        t: Optional[Union[int, float]] = None,
        weighted: bool = False,
    ) -> Dict[str, List[float]]:
        """
        Computes the ROC-AUC score when the numerical variables are turned into binary using a certain threshold.

        Args:
            var1: The first variable to use in the analysis; it has to be a column in the targeting dataset.
            var2: The second variable to use in the analysis; it has to be a column in the targeting dataset.
            p: The percentile to use to convert numerical values into binary.
            t: The threshold to use to convert numerical values into binary.
            weighted: If True the weighted version of the dataset will be used.

        Returns: A dict with AUC score and false/true positive rate.
        """
        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting
        results = {}

        # If threshold is provided, convert to percentile
        p = self.threshold_to_percentile(p, t, data, var1)

        # Stack var1 and var2 together
        a = np.vstack(
            [
                data[var1].astype("float").values.flatten(),
                data[var2].astype("float").values.flatten(),
            ]
        )

        # Order by var1, assign targeting
        num_ones = int((p / 100) * len(data))
        num_zeros = len(data) - num_ones
        targeting_vector = np.concatenate([np.ones(num_ones), np.zeros(num_zeros)])
        a = a[:, a[0, :].argsort()]
        a[0, :] = targeting_vector

        # Record AUC, FPR grid, and TPR grid
        results["auc"] = roc_auc_score(a[0, :], -a[1, :])
        roc = roc_curve(a[0, :], -a[1, :])
        results["fpr"] = roc[0]
        results["tpr"] = roc[1]

        return results

    def auc_overall(
        self, var1: str, var2: str, weighted: bool = False, n_grid: int = 99
    ) -> Dict[str, List[float]]:
        """
        Computed the ROC-AUC scores at 'n_grid' equally spaced percentile thresholds.

        Args:
            var1: The first variable to use in the analysis; it has to be a column in the targeting dataset.
            var2: The second variable to use in the analysis; it has to be a column in the targeting dataset.
            weighted: If True the weighted version of the dataset will be used.
            n_grid: The number of percentiles to compute the AUC scores for.

        Returns: A dict with false/true positive rates and AUC scores for 'n_grid' percentiles between 0 and 100.
        """
        # Generate grid of thresholds between 0 and 100
        grid = np.linspace(1, 100, n_grid)[:-1]

        # Get TPR and FPR for each threshold in grid
        metrics_grid = [
            self.binary_metrics(var1, var2, p, p, weighted=weighted) for p in grid
        ]
        tprs, fprs = [g["tpr"] for g in metrics_grid], [g["fpr"] for g in metrics_grid]

        # When threshold is 0, FPR + TPR are always 0; when threshold is 100, FPR + TPR are always 100
        fprs = [0.0] + fprs + [1]
        tprs = [0.0] + tprs + [1]

        # Smooth any non-increasing parts of the vectors so we can calculate the AUC
        while not strictly_increasing(fprs):
            to_remove = []
            for j in range(1, len(fprs)):
                if fprs[j] <= fprs[j - 1]:
                    to_remove.append(j)
            fprs = [fprs[i] for i in range(len(fprs)) if i not in to_remove]
            tprs = [tprs[i] for i in range(len(tprs)) if i not in to_remove]

        # Calculate the AUC score
        return {"fpr": fprs, "tpr": tprs, "auc": auc(fprs, tprs)}

    def utility(
        self,
        var1: str,
        var2: str,
        transfer_size: Union[float, int],
        p: Optional[Union[int, float]],
        t: Optional[Union[int, float]] = None,
        weighted: bool = False,
        rho: Union[int, float] = 3,
    ) -> float:
        """
        Computes the constant relative risk-aversion (CRRA) [Hanna & Olken (2018)] utility when P% of the population is
        targeted and they receive transfers of size 'transfer_size'.

        Args:
            var1: The first variable to use in the analysis; it has to be a column in the targeting dataset.
            var2: The second variable to use in the analysis; it has to be a column in the targeting dataset.
            transfer_size: The size of the cash transfer.
            p: The percentile to use to convert numerical values into binary.
            t: The threshold to use to convert numerical values into binary.
            weighted: If True the weighted version of the dataset will be used.
            rho: The rho parameter in the constant relative risk-aversion (CRRA) utility function.

        Returns: The utility obtained by targeting that fraction of the population with that transfer size.

        """
        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting
        data = data[[var1, var2]].sort_values(var2, ascending=True)

        # If threshold is provided, convert to percentile
        p = self.threshold_to_percentile(p, t, data, var1, test_percentiles=False)

        # Calculate who is targeted
        num_targeted = int((p / 100) * len(data))
        data["targeted"] = np.concatenate(
            [np.ones(num_targeted), np.zeros(len(data) - num_targeted)]
        )

        # Calculate total utility based on ground truth poverty and benefits
        data["benefits"] = data["targeted"] * transfer_size
        data["utility"] = ((data[var1] + data["benefits"]) ** (1 - rho)) / (1 - rho)

        return data["utility"].sum()

    def targeting_table(
        self,
        groundtruth: str,
        proxies: List[str],
        p1: Optional[Union[int, float]],
        p2: Optional[Union[int, float]],
        t1: Optional[Union[int, float]] = None,
        weighted: bool = False,
    ) -> PandasDataFrame:
        """
        Produces a table with the following metrics for all targeting methods specified by 'proxies': Pearson and
        Spearman correlation, threshold-specific and threshold-agnostic AUC, accuracy, precision, and recall.

        Args:
            groundtruth: The name of the groundtruth column in the targeting dataset.
            proxies: The list of targeting methods to be compared against each other; each name should have a
                corresponding column in the targeting dataset.
            p1: The percentile threshold to use for the groundtruth variable.
            p2: The percentile threshold to use for the proxy variables.
            t1: The threshold to use for the groundtruth variable.
            weighted: If True the weighted version of the dataset will be used.

        Returns: The pandas dataframe with all the performance metrics as columns and the proxies as rows.
        """
        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting

        # If thresholds are provided, convert to percentiles
        p1 = self.threshold_to_percentile(p1, t1, data, groundtruth)

        # Results table: Spearman, Pearson, AUC (Threshold-Agnostic),
        # Accuracy, Precision, Recall, AUC (Threshold-Specific)
        results = pd.DataFrame()
        results["Targeting Method"] = proxies
        results["Pearson"] = [
            self.pearson(groundtruth, proxy, weighted=weighted) for proxy in proxies
        ]
        results["Spearman"] = [
            self.spearman(groundtruth, proxy, weighted=weighted) for proxy in proxies
        ]
        results["AUC (Threshold-Agnostic)"] = [
            self.auc_overall(groundtruth, proxy, weighted=weighted)["auc"]
            for proxy in proxies
        ]
        metrics = [
            self.binary_metrics(groundtruth, proxy, p1, p2, weighted=weighted)
            for proxy in proxies
        ]
        results["Accuracy"] = [m["accuracy"] for m in metrics]
        results["Precision"] = [m["precision"] for m in metrics]
        results["Recall"] = [m["recall"] for m in metrics]
        auc_specifics = [
            self.auc_threshold(groundtruth, proxy, p1, weighted=weighted)
            for proxy in proxies
        ]
        results["AUC (Threshold-Specific)"] = [
            auc_specific["auc"] for auc_specific in auc_specifics
        ]

        # Save and return results
        pecentile_label = (
            "_percentile1=" + str(p1) + "%" + "_percentile2=" + str(p2) + "%"
        )
        results.to_csv(
            self.outputs + "/targeting_table_" + pecentile_label + ".csv", index=False
        )
        return results

    def roc_curves(
        self,
        groundtruth: str,
        proxies: List[str],
        p: Optional[Union[int, float]] = None,
        t: Optional[Union[int, float]] = None,
        weighted: bool = False,
        colors: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Plots the ROC curves for all targeting methods specified by 'proxies', using the percentile or threshold
        provided.

        Args:
            groundtruth: The name of the groundtruth column in the targeting dataset.
            proxies: The list of targeting methods to be compared against each other; each name should have a
                corresponding column in the targeting dataset.
            p: The percentile to use to convert numerical values into binary.
            t: The threshold to use to convert numerical values into binary.
            weighted: If True the weighted version of the dataset will be used.
            colors: Dict mapping proxy names to the color to use (in the default color palette)
        """
        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting

        # If threshold is provided, convert to percentile
        if p is not None or t is not None:
            p = self.threshold_to_percentile(p, t, data, groundtruth)

        # If threshold/percentile not provided, use threshold-agnostic ROC. If threshold/percentile is provided,
        # use specific AUC.
        if p is None:
            rocs = {
                proxy: self.auc_overall(groundtruth, proxy, weighted=weighted)
                for proxy in proxies
            }
        else:
            rocs = {
                proxy: self.auc_threshold(groundtruth, proxy, p, weighted=weighted)
                for proxy in proxies
            }

        # Set up color palette
        if colors is None:
            colors = self.default_colors
        colors_proxy = {proxy: colors[p] for p, proxy in enumerate(proxies)}

        # Plot ROC curves
        fix, ax = plt.subplots(1, figsize=(10, 8))
        for proxy in proxies:
            ax.plot(
                [100 * x for x in rocs[proxy]["fpr"]],
                [100 * x for x in rocs[proxy]["tpr"]],
                label=proxy,
                color=colors_proxy[proxy],
            )
        ax.plot([0, 100], [0, 100], color="grey", dashes=[1, 1], label="Random")

        # Clean up plot
        plt.legend(loc="best")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlim(0, 102)
        ax.set_ylim(0, 102)
        if p is None:
            ax.set_title("Threshold-Agnostic ROC Curves")
        else:
            ax.set_title("ROC Curves: Threshold=" + str(p) + "%")
        clean_plot(ax)

        # Save and show plot
        pecentile_label = "_percentile=" + str(p) + "%" if p is not None else "_overall"
        plt.savefig(self.outputs + "/auc_curves" + pecentile_label + ".png", dpi=400)
        plt.show()

    def precision_recall_curves(
        self,
        groundtruth: str,
        proxies: List[str],
        p: Optional[Union[int, float]] = None,
        t: Optional[Union[int, float]] = None,
        weighted: bool = False,
        n_grid: int = 99,
        colors: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Plots the ROC curves for all targeting methods specified by 'proxies', using the percentile/threshold
        provided OR a set of 'n_grid' equally spaced percentiles.

        Args:
            groundtruth: The name of the groundtruth column in the targeting dataset.
            proxies: The list of targeting methods to be compared against each other; each name should have a
                corresponding column in the targeting dataset.
            p: The percentile to use to convert numerical values into binary.
            t: The threshold to use to convert numerical values into binary.
            weighted: If True the weighted version of the dataset will be used.
            n_grid: The number of equally-spaced percentiles to compute the metrics for.
            colors: Color palette to use.
        """
        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting

        # If threshold is provided, convert to percentile
        if p is not None or t is not None:
            p = self.threshold_to_percentile(p, t, data, groundtruth)

        # If threshold/percentile is not provided, use balanced grid. Otherwise, use threshold provided for
        # threshold-specific measures.
        grid = np.linspace(1, 100, n_grid)[:-1]
        if p is None:
            metrics_grid = {
                proxy: [
                    self.binary_metrics(groundtruth, proxy, p2, p2, weighted=weighted)
                    for p2 in grid
                ]
                for proxy in proxies
            }
        else:
            metrics_grid = {
                proxy: [
                    self.binary_metrics(groundtruth, proxy, p, p2, weighted=weighted)
                    for p2 in grid
                ]
                for proxy in proxies
            }

        # Set up color palette
        if colors is None:
            colors = self.default_colors
        colors_proxy = {proxy: colors[p] for p, proxy in enumerate(proxies)}

        # Plot precision and recall curves (only precision if using balanced grid, since precision and recall are the
        # same)
        fig, ax = plt.subplots(1, figsize=(10, 8))
        for proxy in proxies:
            ax.plot(
                grid,
                [100 * metrics_grid[proxy][i]["precision"] for i in range(len(grid))],
                label=proxy,
                color=colors_proxy[proxy],
            )
            if p is not None:
                ax.plot(
                    grid,
                    [100 * metrics_grid[proxy][i]["recall"] for i in range(len(grid))],
                    dashes=[2, 2],
                    color=colors_proxy[proxy],
                )

        # Clean up plot
        plt.legend(loc="best")
        ax.set_xlabel("Share of Population Targeted")
        if p is None:
            ax.set_ylabel("Precision and Recall")
        else:
            ax.set_ylabel("Precision (Solid) and Recall (Dashed)")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlim(0, 102)
        ax.set_ylim(0, 102)
        if p is None:
            ax.set_title("Threshold-Agnostic Precision-Recall Curves")
        else:
            ax.set_title("Precision-Recall Curves: Threshold=" + str(int(p)) + "%")
        clean_plot(ax)

        # Save and show plot
        pecentile_label = "_percentile=" + str(p) + "%" if p is not None else "_overall"
        plt.savefig(
            self.outputs + "/precision_recall_curves" + pecentile_label + ".png",
            dpi=400,
        )
        plt.show()

    def utility_grid(
        self,
        groundtruth: str,
        proxies: List[str],
        ubi_transfer_size: Union[int, float],
        weighted: bool = False,
        n_grid: int = 99,
    ) -> Tuple[np.ndarray, List[float], Dict[str, List[float]]]:
        """
        Computes the utility achieved by all targeting methods specified by 'proxies', when targeting 'n_grid' equally
        spaced fractions of the population.

        Args:
            groundtruth: The name of the groundtruth column in the targeting dataset.
            proxies: The list of targeting methods to be compared against each other; each name should have a
                corresponding column in the targeting dataset.
            ubi_transfer_size: The size of the cash transfer if the entire population was targeted.
            weighted: If True the weighted version of the dataset will be used.
            n_grid: The number of equally-spaced percentiles to compute the metrics for.
        """
        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting

        # Set up grid and results
        budget = ubi_transfer_size * len(data)
        grid = np.linspace(1, 100, n_grid)
        utilities: Dict[str, List[float]] = {proxy: [] for proxy in proxies}
        transfer_sizes = []

        # Calculate utility and transfer size for each proxy and each value in grid
        for p in grid:
            num_targeted = int(len(data) * (p / 100))
            transfer_size = budget / num_targeted
            transfer_sizes.append(transfer_size)
            for proxy in proxies:
                utilities[proxy] = utilities[proxy] + [
                    self.utility(
                        groundtruth, proxy, transfer_size, p, weighted=weighted
                    )
                ]

        return grid, transfer_sizes, utilities

    def utility_curves(
        self,
        groundtruth: str,
        proxies: List[str],
        ubi_transfer_size: Union[int, float],
        weighted: bool = False,
        n_grid: int = 99,
        colors: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Plots the utility curves of all targeting methods specified by 'proxies', when targeting 'n_grid' equally
        spaced fractions of the population.

        Args:
            groundtruth: The name of the groundtruth column in the targeting dataset.
            proxies: The list of targeting methods to be compared against each other; each name should have a
                corresponding column in the targeting dataset.
            ubi_transfer_size: The size of the cash transfer if the entire population was targeted.
            weighted: If True the weighted version of the dataset will be used.
            n_grid: The number of equally-spaced percentiles to compute the metrics for.
            colors: Color palette to use.
        """
        # Get utility grid
        grid, _, utilities = self.utility_grid(
            groundtruth, proxies, ubi_transfer_size, weighted=weighted, n_grid=n_grid
        )

        # Set up color palette
        if colors is None:
            colors = self.default_colors
        colors_proxy = {proxy: colors[p] for p, proxy in enumerate(proxies)}

        # Plot utility curves for each proxy
        fix, ax = plt.subplots(1, figsize=(10, 8))
        for proxy in proxies:
            ax.plot(grid, utilities[proxy], color=colors_proxy[proxy], label=proxy)

        # Plot circle for maximum utility for each proxy
        for proxy in proxies:
            max_utility_idx = np.argmax(utilities[proxy])
            ax.scatter(
                [grid[max_utility_idx]],
                [utilities[proxy][max_utility_idx]],
                color=colors_proxy[proxy],
                s=80,
            )

        # Add horizontal line for UBI
        ax.axhline(utilities[proxies[0]][-1], color="grey", dashes=[1, 1], label="UBI")

        # Clean up plot
        plt.legend(loc="upper right")
        ax.set_xlabel("Share of Population Targeted")
        ax.set_ylabel("Utility (CRRA Assumptions)")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlim(0, 102)
        ax.set_title("Utility Curves")
        clean_plot(ax)

        # Save and show plot
        plt.savefig(self.outputs + "/utility_curves.png", dpi=400)
        plt.show()

    def utility_table(
        self,
        groundtruth: str,
        proxies: List[str],
        ubi_transfer_size: Union[int, float],
        weighted: bool = False,
        n_grid: int = 99,
    ) -> PandasDataFrame:
        """
        Computes the optimal share of the population targeted, maximum utility, and optimal transfer size of all
        targeting methods specified by 'proxies', out of 'n_grid' equally spaced fractions of the population.

        Args:
            groundtruth: The name of the groundtruth column in the targeting dataset.
            proxies: The list of targeting methods to be compared against each other; each name should have a
                corresponding column in the targeting dataset.
            ubi_transfer_size: The size of the cash transfer if the entire population was targeted.
            weighted: If True the weighted version of the dataset will be used.
            n_grid: The number of equally-spaced percentiles to compute the metrics for.

        Returns: The pandas dataframe with optimal utility metrics for each targeting method.
        """
        # Get utility grid and transfer size grid
        population_shares, transfer_sizes, utilities = self.utility_grid(
            groundtruth, proxies, ubi_transfer_size, weighted=weighted, n_grid=n_grid
        )

        # Create table with best population share, utility, and transfer size for each proxy
        table_population_shares, table_utilities, table_transfer_sizes = [], [], []
        for proxy in proxies:
            max_utility_idx = np.argmax(utilities[proxy])
            table_population_shares.append(population_shares[max_utility_idx])
            table_utilities.append(utilities[proxy][max_utility_idx])
            table_transfer_sizes.append(transfer_sizes[max_utility_idx])

        # Clean up table
        table = pd.DataFrame()
        table["Proxy"] = proxies
        table["Optimal Share of Population Targeted"] = table_population_shares
        table["Maximum Utility"] = table_utilities
        table["Optimal Transfer Size"] = table_transfer_sizes

        # Save and return table
        table.to_csv(self.outputs + "/optimal_utility_table.csv", index=False)
        return table
