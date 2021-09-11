# TODO: parallelize lasso and forward selection
from box import Box
import yaml
from helpers.utils import *
from helpers.plot_utils import *
from helpers.io_utils import *
from helpers.ml_utils import *
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve, auc


class Targeting:

    def __init__(self, cfg_dir, dataframe=None, clean_folders=False):

        # Read config file
        with open(cfg_dir, "r") as ymlfile:
            cfg = Box(yaml.safe_load(ymlfile))
        self.cfg = cfg
        data_path = cfg.path.targeting.data + cfg.path.targeting.file_names.data
        self.data = pd.read_csv(data_path)
        self.data['random'] = np.random.rand(len(self.data))
        outputs = cfg.path.targeting.outputs
        self.outputs = outputs
        file_names = cfg.path.targeting.file_names
        self.default_colors = sns.color_palette('Set2', 100)

        # Unweighted data
        self.unweighted_data = self.data.copy()
        self.unweighted_data['weight'] = 1

        # Weighted data
        self.weighted_data = self.data.copy()
        if 'weight' not in self.weighted_data.columns:
            self.data['weight'] = 1
        else:
            self.weighted_data['weight'] = self.weighted_data['weight']/self.weighted_data['weight'].min()
        self.weighted_data = pd.DataFrame(np.repeat(self.weighted_data.values, self.weighted_data['weight'], axis=0), columns=self.weighted_data.columns)

    def pearson(self, var1, var2, weighted=False):
        data = self.weighted_data if weighted else self.unweighted_data
        return np.corrcoef(data[var1], data[var2])[0][1]
    
    def spearman (self, var1, var2, weighted=False):
        data = self.weighted_data if weighted else self.unweighted_data
        return spearmanr(data[var1], data[var2])[0]

    def binary_metrics(self, var1, var2, p1, p2, weighted=False):

        data = self.weighted_data if weighted else self.unweighted_data
        results = {}

        if p1 == 0 or p2 == 0 or p1 == 100 or p2 == 100:
            raise ValueError('Pecentage targeting must be between 0 and 100 (exclusive).')

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
    
    def auc_threshold(self, var1, var2, p, weighted=False):

        data = self.weighted_data if weighted else self.unweighted_data
        results = {}

        if p == 0 or p == 100:
            raise ValueError('Pecentage targeting must be between 0 and 100 (exclusive).')

        # Stack var1 and var2 together
        a = np.vstack([data[var1].values.flatten(), data[var2].values.flatten()])

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

    
    def auc_overall(self, var1, var2, weighted=False, n_grid=99):

        data = self.weighted_data if weighted else self.unweighted_data

        # Generate grid of thresholds between 0 and 100
        grid = np.linspace(1, 100, n_grid)[:-1]

        # Get TPR and FPR for each threshold in grid
        metrics_grid = [self.binary_metrics(var1, var2, p, p, weighted=weighted) for p in grid]
        tprs, fprs = [g['tpr'] for g in metrics_grid], [g['fpr'] for g in metrics_grid]
        
        # When threshold is 0, FPR + TPR are always 0; when threshold is 100, FPR + TPR are always 100 
        fprs = [0] + fprs + [1]
        tprs = [0] + tprs + [1]
        
        # Smooth any non-increasing parts of the vectors so we can calculate the AUC 
        while not strictly_increasing(fprs):
            to_remove = []
            for j in range(1, len(fprs)):
                if fprs[j] <= fprs[j-1]:
                    to_remove.append(j)
            fprs = [fprs[i] for i in range(len(fprs)) if i not in to_remove]
            tprs = [tprs[i] for i in range(len(tprs)) if i not in to_remove]
        
        # Calculate the AUC score 
        return {'fpr':fprs, 'tpr':tprs, 'auc':auc(fprs, tprs)} 

    def utility(self, var1, var2, p, transfer_size, weighted=False, rho=3):

        data = self.weighted_data if weighted else self.unweighted_data
        data = data[[var1, var2]].sort_values(var2, ascending=True)

        num_targeted = int((p/100)*len(data))
        data['targeted'] = np.concatenate([np.ones(num_targeted), np.zeros(len(data) - num_targeted)])
        data['benefits'] = data['targeted']*transfer_size
        data['utility'] = ((data[var1] + data['benefits'])**(1-rho))/(1-rho)
        return data['utility'].sum()

    
    def targeting_table(self, groundtruth, proxies, p1, p2, weighted=False):
        results = pd.DataFrame()
        results['Targeting Method'] = proxies 
        results['Pearson'] = [self.pearson(groundtruth, proxy, weighted=weighted) for proxy in proxies]
        results['Spearman'] = [self.spearman(groundtruth, proxy, weighted=weighted) for proxy in proxies]
        results['AUC (Threshold-Agnostic)'] = [self.auc_overall(groundtruth, proxy, weighted=weighted)['auc'] for proxy in proxies]
        metrics = [self.binary_metrics(groundtruth, proxy, p1, p2, weighted=weighted) for proxy in proxies]
        results['Accuracy'] = [m['accuracy'] for m in metrics]
        results['Precision'] = [m['precision'] for m in metrics]
        results['Recall'] = [m['recall'] for m in metrics]
        auc_specifics = [self.auc_threshold(groundtruth, proxy, p1, weighted=weighted) for proxy in proxies]
        results['AUC (Threshold-Specific)'] = [auc_specific['auc'] for auc_specific in auc_specifics]
        return results

    def roc_curves(self, groundtruth, proxies, p=None, weighted=False, colors=None):

        if colors is None:
            colors = self.default_colors
        colors = {proxy: colors[p] for p, proxy in enumerate(proxies)}

        # If threshold not provided, use threshold-agnostic ROC
        if p is None:
            rocs = {proxy:self.auc_overall(groundtruth, proxy, weighted=weighted) for proxy in proxies}
        else:
            rocs = {proxy:self.auc_threshold(groundtruth, proxy, p, weighted=weighted) for proxy in proxies}

        fix, ax = plt.subplots(1, figsize=(10, 8))
        for proxy in proxies:
            ax.plot([100*x for x in rocs[proxy]['fpr']], [100*x for x in rocs[proxy]['tpr']], label=proxy, color=colors[proxy])
        
        ax.plot([0, 100], [0, 100], color='grey', dashes=[1, 1], label='Random')
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
        plt.show()

    def precision_recall_curves(self, groundtruth, proxies, p=None, weighted=False, n_grid=99, colors=None):

        if colors is None:
            colors = self.default_colors
        colors = {proxy: colors[p] for p, proxy in enumerate(proxies)}

        grid = np.linspace(1, 100, n_grid)[:-1]

        if p is None:
            metrics_grid = {proxy:[self.binary_metrics(groundtruth, proxy, p2, p2, weighted=weighted) for p2 in grid] for proxy in proxies}
        else:
            metrics_grid = {proxy:[self.binary_metrics(groundtruth, proxy, p, p2, weighted=weighted) for p2 in grid] for proxy in proxies}

        fig, ax = plt.subplots(1, figsize=(10, 8))
        for proxy in proxies:
            ax.plot(grid, [100*metrics_grid[proxy][i]['precision'] for i in range(len(grid))], label=proxy, color=colors[proxy])
            if p is not None:
                ax.plot(grid, [100*metrics_grid[proxy][i]['recall'] for i in range(len(grid))], dashes=[2, 2], color=colors[proxy])
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
            ax.set_title('Precision-Recall Curves: Threshold=' + str(p) + '%')
        clean_plot(ax)
        plt.show()

    def utility_grid(self, groundtruth, proxies, ubi_transfer_size, weighted=False, n_grid=99):
        data = self.weighted_data if weighted else self.unweighted_data
        budget = ubi_transfer_size*len(data)
        grid = np.linspace(1, 100, n_grid)
        utilities = {proxy:[] for proxy in proxies}
        transfer_sizes = []
        for p in grid:
            num_targeted = int(len(data)*(p/100))
            transfer_size = budget/num_targeted
            transfer_sizes.append(transfer_size)
            for proxy in proxies:
                utilities[proxy] = utilities[proxy] + [self.utility(groundtruth, proxy, p, transfer_size, weighted=weighted)]
        return grid, transfer_sizes, utilities


    def utility_curves(self, groundtruth, proxies, ubi_transfer_size, weighted=False, n_grid=99, colors=None):

        if colors is None:
            colors = self.default_colors
        colors = {proxy: colors[p] for p, proxy in enumerate(proxies)}

        grid, _, utilities = self.utility_grid(groundtruth, proxies, ubi_transfer_size, weighted=weighted, n_grid=n_grid)
        
        fix, ax = plt.subplots(1, figsize=(10, 8))
        for proxy in proxies:
            ax.plot(grid, utilities[proxy], color=colors[proxy], label=proxy)
        for proxy in proxies:
            max_utility_idx = np.argmax(utilities[proxy])
            ax.scatter([grid[max_utility_idx]], [utilities[proxy][max_utility_idx]], color=colors[proxy], s=80)
        ax.axhline(utilities[proxies[0]][-1], color='grey', dashes=[1, 1], label='UBI')
        plt.legend(loc='upper right')
        ax.set_xlabel('Share of Population Targeted')
        ax.set_ylabel('Utility (CRRA Assumptions)')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlim(0, 102)
        ax.set_title('Utility Curves')
        clean_plot(ax)
        plt.show()

    def utility_table(self, groundtruth, proxies, ubi_transfer_size, weighted=False, n_grid=99):

        population_shares, transfer_sizes, utilities = self.utility_grid(groundtruth, proxies, ubi_transfer_size, weighted=weighted, n_grid=n_grid)
        
        table_population_shares, table_utilities, table_transfer_sizes = [], [], []
        for proxy in proxies:
            max_utility_idx = np.argmax(utilities[proxy])
            table_population_shares.append(population_shares[max_utility_idx])
            table_utilities.append(utilities[proxy][max_utility_idx])
            table_transfer_sizes.append(transfer_sizes[max_utility_idx])
        
        table = pd.DataFrame()
        table['Proxy'] = proxies
        table['Optimal Share of Population Targeted'] = table_population_shares
        table['Maximum Utility'] = table_utilities
        table['Optimal Transfer Size'] = table_transfer_sizes
        
        return table



        



            


















