# TODO: parallelize lasso and forward selection
from datastore import DataStore, DataType
from helpers.plot_utils import clean_plot, mtick
from helpers.ml_utils import confusion_matrix, strictly_increasing
from helpers.utils import make_dir
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd
from pandas import DataFrame as PandasDataFrame
from scipy.stats import spearmanr  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
from sklearn.metrics import roc_auc_score, roc_curve, auc  # type: ignore[import]
from typing import Dict, List, Optional, Tuple, Union


class Targeting:

    def __init__(self, datastore: DataStore, clean_folders: bool = False) -> None:
        self.cfg = datastore.cfg
        self.ds = datastore
        self.outputs = datastore.outputs + 'featurizer/'

        # Prepare working directories, set color palette
        make_dir(self.outputs, clean_folders)
        self.default_colors = sns.color_palette('Set2', 100)

        # Load data into datastore
        data_type_map = {DataType.TARGETING: None}
        self.ds.load_data(data_type_map=data_type_map)

    @staticmethod
    def threshold_to_percentile(p: Optional[Union[float, int]],
                                t: Optional[Union[float, int]],
                                data: PandasDataFrame,
                                var: str,
                                test_percentiles: bool = True) -> Union[float, int]:

        # Make sure at least one of t or p is provided
        if p is None and t is None:
            raise ValueError('Must provide percentage targeting or threshold targeting')

        # Make sure not both t and p are provided
        if p is not None and t is not None:
            raise ValueError('Cannot provide both percentage and threshold targeting for var1')

        # Make sure percentile is within range (0 to 100 exclusive)
        if test_percentiles and (p == 0 or p == 100):
            raise ValueError('Percentage targeting must be between 0 and 100 (exclusive).')
        
        # If t is provided, convert threshold to percentile
        if t is not None:
            p = 100*(len(data[data[var] < t])/len(data))
        
        return p

    def pearson(self, var1: str, var2: str, weighted: bool = False) -> float:
        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting
        return np.corrcoef(data[var1].astype('float'), data[var2].astype('float'))[0][1]

    def spearman(self, var1: str, var2: str, weighted: bool = False) -> float:
        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting
        return spearmanr(data[var1].astype('float'), data[var2].astype('float'))[0]

    def binary_metrics(self, var1: str, var2: str,
                       p1: Union[int, float], p2: Optional[Union[int, float]],
                       t1: Optional[Union[int, float]] = None, t2: Optional[Union[int, float]] = None,
                       weighted: bool = False) -> Dict[str, float]:

        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting
        results = {}

        # If thresholds are provided, convert to percentiles
        p1 = self.threshold_to_percentile(p1, t1, data, var1)
        p2 = self.threshold_to_percentile(p2, t2, data, var2)

        # Stack var1 and var2 together
        a = np.vstack([data[var1].values.flatten(), data[var2].values.flatten()])

        # Order by var1, assign targeting
        num_ones = int((p1/100)*len(data))
        num_zeros = len(data) - num_ones
        targeting_vector = np.concatenate([np.ones(num_ones), np.zeros(num_zeros)])
        a = a[:, a[0, :].argsort()]
        a[0, :] = targeting_vector

        # Reshuffle
        np.random.seed(1)
        a = a[:, np.random.rand(a.shape[1]).argsort()]
    
        # Order by var2, assign targeting
        num_ones = int((p2/100)*len(data))
        num_zeros = len(data) - num_ones
        targeting_vector = np.concatenate([np.ones(num_ones), np.zeros(num_zeros)])
        a = a[:, a[1, :].argsort()]
        a[1, :] = targeting_vector

        # Calculate confusion matrix and from there the binary metrics
        tn, fp, fn, tp = confusion_matrix(a[0, :], a[1, :]).ravel()
        results['accuracy'] = (tp + tn)/(tp + tn + fp + fn)
        results['precision'] = tp/(tp + fp)
        results['recall'] = tp/(tp + fn)
        results['tpr'] = tp/(tp + fn)
        results['fpr'] = fp/(fp + tn)
        return results

    def auc_threshold(self, var1: str, var2: str,
                      p: Optional[Union[int, float]], t: Optional[Union[int, float]] = None,
                      weighted: bool = False) -> Dict[str, List[float]]:

        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting
        results = {}

        # If threshold is provided, convert to percentile
        p = self.threshold_to_percentile(p, t, data, var1)

        # Stack var1 and var2 together
        a = np.vstack([data[var1].astype('float').values.flatten(), data[var2].astype('float').values.flatten()])

        # Order by var1, assign targeting
        num_ones = int((p/100)*len(data))
        num_zeros = len(data) - num_ones
        targeting_vector = np.concatenate([np.ones(num_ones), np.zeros(num_zeros)])
        a = a[:, a[0, :].argsort()]
        a[0, :] = targeting_vector

        # Record AUC, FPR grid, and TPR grid
        results['auc'] = roc_auc_score(a[0, :], -a[1, :])
        roc = roc_curve(a[0, :], -a[1, :])
        results['fpr'] = roc[0]
        results['tpr'] = roc[1]

        return results

    def auc_overall(self, var1: str, var2: str,
                    weighted: bool = False, n_grid: int = 99) -> Dict[str, List[float]]:
        # Generate grid of thresholds between 0 and 100
        grid = np.linspace(1, 100, n_grid)[:-1]

        # Get TPR and FPR for each threshold in grid
        metrics_grid = [self.binary_metrics(var1, var2, p, p, weighted=weighted) for p in grid]
        tprs, fprs = [g['tpr'] for g in metrics_grid], [g['fpr'] for g in metrics_grid]
        
        # When threshold is 0, FPR + TPR are always 0; when threshold is 100, FPR + TPR are always 100 
        fprs = [0.] + fprs + [1]
        tprs = [0.] + tprs + [1]
        
        # Smooth any non-increasing parts of the vectors so we can calculate the AUC 
        while not strictly_increasing(fprs):
            to_remove = []
            for j in range(1, len(fprs)):
                if fprs[j] <= fprs[j-1]:
                    to_remove.append(j)
            fprs = [fprs[i] for i in range(len(fprs)) if i not in to_remove]
            tprs = [tprs[i] for i in range(len(tprs)) if i not in to_remove]
        
        # Calculate the AUC score 
        return {'fpr': fprs, 'tpr': tprs, 'auc': auc(fprs, tprs)}

    def utility(self, var1: str, var2: str,
                transfer_size: Union[float, int],
                p: Optional[Union[int, float]], t: Optional[Union[int, float]] = None,
                weighted: bool = False, rho: Union[int, float] = 3) -> float:

        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting
        data = data[[var1, var2]].sort_values(var2, ascending=True)

        # If threshold is provided, convert to percentile
        p = self.threshold_to_percentile(p, t, data, var1, test_percentiles=False)

        # Calculate who is targeted
        num_targeted = int((p/100)*len(data))
        data['targeted'] = np.concatenate([np.ones(num_targeted), np.zeros(len(data) - num_targeted)])

        # Calculate total utility based on ground truth poverty and benefits
        data['benefits'] = data['targeted']*transfer_size
        data['utility'] = ((data[var1] + data['benefits'])**(1-rho))/(1-rho)

        return data['utility'].sum()
    
    def targeting_table(self, groundtruth: str, proxies: List[str],
                        p1: Optional[Union[int, float]], p2: Optional[Union[int, float]],
                        t1: Optional[Union[int, float]] = None, weighted: bool = False) -> PandasDataFrame:

        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting

        # If thresholds are provided, convert to percentiles
        p1 = self.threshold_to_percentile(p1, t1, data, groundtruth)

        # Results table: Spearman, Pearson, AUC (Threshold-Agnostic),
        # Accuracy, Precision, Recall, AUC (Threshold-Specific)
        results = pd.DataFrame()
        results['Targeting Method'] = proxies 
        results['Pearson'] = [self.pearson(groundtruth, proxy, weighted=weighted) for proxy in proxies]
        results['Spearman'] = [self.spearman(groundtruth, proxy, weighted=weighted) for proxy in proxies]
        results['AUC (Threshold-Agnostic)'] = [self.auc_overall(groundtruth, proxy, weighted=weighted)['auc']
                                               for proxy in proxies]
        metrics = [self.binary_metrics(groundtruth, proxy, p1, p2, weighted=weighted) for proxy in proxies]
        results['Accuracy'] = [m['accuracy'] for m in metrics]
        results['Precision'] = [m['precision'] for m in metrics]
        results['Recall'] = [m['recall'] for m in metrics]
        auc_specifics = [self.auc_threshold(groundtruth, proxy, p1, weighted=weighted) for proxy in proxies]
        results['AUC (Threshold-Specific)'] = [auc_specific['auc'] for auc_specific in auc_specifics]

        # Save and return results
        pecentile_label = '_percentile1=' + str(p1) + '%' + '_percentile2=' + str(p2) + '%'
        results.to_csv(self.outputs + '/targeting_table_' + pecentile_label + '.csv', index=False)
        return results

    def roc_curves(self, groundtruth: str, proxies: List[str],
                   p: Optional[Union[int, float]] = None, t: Optional[Union[int, float]] = None,
                   weighted: bool = False, colors: Optional[Dict[int, str]] = None) -> None:

        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting
        
        # If threshold is provided, convert to percentile
        if p is not None or t is not None:
            p = self.threshold_to_percentile(p, t, data, groundtruth)

        # If threshold/percentile not provided, use threshold-agnostic ROC. If threshold/percentile is provided,
        # use specific AUC.
        if p is None:
            rocs = {proxy: self.auc_overall(groundtruth, proxy, weighted=weighted) for proxy in proxies}
        else:
            rocs = {proxy: self.auc_threshold(groundtruth, proxy, p, weighted=weighted) for proxy in proxies}

        # Set up color palette
        if colors is None:
            colors = self.default_colors
        colors_proxy = {proxy: colors[p] for p, proxy in enumerate(proxies)}

        # Plot ROC curves
        fix, ax = plt.subplots(1, figsize=(10, 8))
        for proxy in proxies:
            ax.plot([100*x for x in rocs[proxy]['fpr']], [100*x for x in rocs[proxy]['tpr']],
                    label=proxy, color=colors_proxy[proxy])
        ax.plot([0, 100], [0, 100], color='grey', dashes=[1, 1], label='Random')

        # Clean up plot
        plt.legend(loc='best')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlim(0, 102)
        ax.set_ylim(0, 102)
        if p is None:
            ax.set_title('Threshold-Agnostic ROC Curves')
        else:
            ax.set_title('ROC Curves: Threshold=' + str(p) + '%')
        clean_plot(ax)
        
        # Save and show plot
        pecentile_label = '_percentile=' + str(p) + '%' if p is not None else '_overall'
        plt.savefig(self.outputs + '/auc_curves' + pecentile_label + '.png', dpi=400)
        plt.show()

    def precision_recall_curves(self, groundtruth: str, proxies: List[str],
                                p: Optional[Union[int, float]] = None, t: Optional[Union[int, float]] = None,
                                weighted: bool = False, n_grid: int = 99,
                                colors: Optional[Dict[int, str]] = None) -> None:

        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting

        # If threshold is provided, convert to percentile
        if p is not None or t is not None:
            p = self.threshold_to_percentile(p, t, data, groundtruth)

        # If threshold/percentile is not provided, use balanced grid. Otherwise, use threshold provided for
        # threshold-specific measures.
        grid = np.linspace(1, 100, n_grid)[:-1]
        if p is None:
            metrics_grid = {proxy: [self.binary_metrics(groundtruth, proxy, p2, p2, weighted=weighted) for p2 in grid]
                            for proxy in proxies}
        else:
            metrics_grid = {proxy: [self.binary_metrics(groundtruth, proxy, p, p2, weighted=weighted) for p2 in grid]
                            for proxy in proxies}

        # Set up color palette
        if colors is None:
            colors = self.default_colors
        colors_proxy = {proxy: colors[p] for p, proxy in enumerate(proxies)}

        # Plot precision and recall curves (only precision if using balanced grid, since precision and recall are the
        # same)
        fig, ax = plt.subplots(1, figsize=(10, 8))
        for proxy in proxies:
            ax.plot(grid, [100*metrics_grid[proxy][i]['precision'] for i in range(len(grid))],
                    label=proxy, color=colors_proxy[proxy])
            if p is not None:
                ax.plot(grid, [100*metrics_grid[proxy][i]['recall'] for i in range(len(grid))],
                        dashes=[2, 2], color=colors_proxy[proxy])
        
        # Clean up plot
        plt.legend(loc='best')
        ax.set_xlabel('Share of Population Targeted')
        if p is None:
            ax.set_ylabel('Precision and Recall')
        else:
            ax.set_ylabel('Precision (Solid) and Recall (Dashed)')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlim(0, 102)
        ax.set_ylim(0, 102)
        if p is None:
            ax.set_title('Threshold-Agnostic Precision-Recall Curves')
        else:
            ax.set_title('Precision-Recall Curves: Threshold=' + str(int(p)) + '%')
        clean_plot(ax)

        # Save and show plot
        pecentile_label = '_percentile=' + str(p) + '%' if p is not None else '_overall'
        plt.savefig(self.outputs + '/precision_recall_curves' + pecentile_label + '.png', dpi=400)
        plt.show()

    def utility_grid(self, groundtruth: str, proxies: List[str],
                     ubi_transfer_size: Union[int, float], weighted: bool = False, n_grid: int = 99
                     ) -> Tuple[np.ndarray, List[float], Dict[str, List[float]]]:

        data = self.ds.weighted_targeting if weighted else self.ds.unweighted_targeting

        # Set up grid and results 
        budget = ubi_transfer_size*len(data)
        grid = np.linspace(1, 100, n_grid)
        utilities: Dict[str, List[float]] = {proxy: [] for proxy in proxies}
        transfer_sizes = []

        # Calculate utility and transfer size for each proxy and each value in grid
        for p in grid:
            num_targeted = int(len(data)*(p/100))
            transfer_size = budget/num_targeted
            transfer_sizes.append(transfer_size)
            for proxy in proxies:
                utilities[proxy] = utilities[proxy] + \
                                   [self.utility(groundtruth, proxy, transfer_size, p, weighted=weighted)]

        return grid, transfer_sizes, utilities

    def utility_curves(self, groundtruth: str, proxies: List[str],
                       ubi_transfer_size: Union[int, float], weighted: bool = False,
                       n_grid: int = 99, colors: Optional[Dict[int, str]] = None) -> None:

        # Get utility grid
        grid, _, utilities = self.utility_grid(groundtruth, proxies, ubi_transfer_size, weighted=weighted,
                                               n_grid=n_grid)

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
            ax.scatter([grid[max_utility_idx]], [utilities[proxy][max_utility_idx]], color=colors_proxy[proxy], s=80)
        
        # Add horizontal line for UBI
        ax.axhline(utilities[proxies[0]][-1], color='grey', dashes=[1, 1], label='UBI')

        # Clean up plot
        plt.legend(loc='upper right')
        ax.set_xlabel('Share of Population Targeted')
        ax.set_ylabel('Utility (CRRA Assumptions)')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlim(0, 102)
        ax.set_title('Utility Curves')
        clean_plot(ax)

        # Save and show plot
        plt.savefig(self.outputs + '/utility_curves.png', dpi=400)
        plt.show()

    def utility_table(self, groundtruth: str, proxies: List[str],
                      ubi_transfer_size: Union[int, float], weighted: bool = False,
                      n_grid: int = 99) -> PandasDataFrame:

        # Get utility grid and transfer size grid
        population_shares, transfer_sizes, utilities = self.utility_grid(groundtruth, proxies, ubi_transfer_size,
                                                                         weighted=weighted, n_grid=n_grid)
        
        # Create table with best population share, utility, and transfer size for each proxy
        table_population_shares, table_utilities, table_transfer_sizes = [], [], []
        for proxy in proxies:
            max_utility_idx = np.argmax(utilities[proxy])
            table_population_shares.append(population_shares[max_utility_idx])
            table_utilities.append(utilities[proxy][max_utility_idx])
            table_transfer_sizes.append(transfer_sizes[max_utility_idx])
        
        # Clean up table
        table = pd.DataFrame()
        table['Proxy'] = proxies
        table['Optimal Share of Population Targeted'] = table_population_shares
        table['Maximum Utility'] = table_utilities
        table['Optimal Transfer Size'] = table_transfer_sizes

        # Save and return table
        table.to_csv(self.outputs + '/optimal_utility_table.csv', index=False)
        return table
