from datastore import DataStore, DataType
from helpers.utils import make_dir
from helpers.plot_utils import clean_plot
from matplotlib.collections import PatchCollection  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd
from pandas import DataFrame as PandasDataFrame
from scipy.stats import f_oneway  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
from typing import Dict, Mapping, List, Optional, Union


class Fairness:

    def __init__(self, datastore: DataStore, clean_folders: bool = False) -> None:
        self.cfg = datastore.cfg
        self.ds = datastore
        self.outputs = datastore.outputs + 'fairness/'

        # Prepare working directories, set color palette
        make_dir(self.outputs, clean_folders)
        self.default_colors = sns.color_palette('Set2', 100)

        # Load data into datastore
        data_type_map = {DataType.FAIRNESS: None}
        self.ds.load_data(data_type_map=data_type_map)

    def rank_residual(self, var1: str, var2: str,
                      characteristic: str, weighted: bool = False) -> Mapping[str, np.ndarray]:
        """
        Returns the distribution of normalized rank-residuals across subgroups and targeting methods.
        The rank-residual is a measure of whether certain groups are consistently ranked higher or lower than they
        "should" be, as defined by Aiken et al. (2021).

        Args:
            var1: The name of the column containing data coming from the first targeting method; can also be the ground-
            truth column.
            var2: The name of the column containing data coming from the second targeting method.
            characteristic: The (demographic) characteristic by which to disaggregate results - this should be the name
            of the corresponding column in the targeting dataset.
            weighted: If True the weighted version of the dataset will be used.

        Returns:
            A dict mapping the characteristic's subgroups to their distributions of normalized rank-residuals.
        """

        data = self.ds.weighted_fairness if weighted else self.ds.unweighted_fairness

        # Get rankings according to var1 and var2
        data = data.sort_values([var1, 'random'], ascending=True)
        data['rank1'] = range(len(data))
        data = data.sort_values([var2, 'random'], ascending=True)
        data['rank2'] = range(len(data))

        # Calculate normalized rank residual
        data['rank_residual'] = (data['rank2'] - data['rank1']) / data['rank2'].max()

        # Return rank residual distributions for each group
        return {group: data[data[characteristic] == group]['rank_residual'].values.flatten()
                for group in data[characteristic].unique()}

    def demographic_parity(self, var1: str, var2: str, characteristic: str,
                           p: Union[float, int], weighted: bool = False
                           ) -> Dict[str, Dict[str, float]]:
        """
        Compares the proportion of each subgroup living in poverty (below the pth percentile in terms of consumption)
        to the proportion of each subgroup that is targeted (below the pth percentile in terms of the proxy poverty
        measure used for targeting) to produce a measure of demographic parity. This is defined as:
        DP = (TP + FP)/N - (TP + FN)/N
        where TP stands for true positives, and so forth.

        Args:
            var1: The name of the column containing data coming from the first targeting method; can also be the ground-
            truth column.
            var2: The name of the column containing data coming from the second targeting method.
            characteristic: The (demographic) characteristic by which to disaggregate results - this should be the name
                of the corresponding column in the targeting dataset.
            p: The percentile below which users will be considered as poor.
            weighted: If True the weighted version of the dataset will be used.

        Returns: A dict mapping the characteristic's subgroups to their poverty shares and demographic parities.
        """

        data = self.ds.weighted_fairness if weighted else self.ds.unweighted_fairness

        # Simulate targeting by var1 and var2
        num_ones = int((p / 100) * len(data))
        num_zeros = len(data) - num_ones
        targeting_vector = np.concatenate([np.ones(num_ones), np.zeros(num_zeros)])
        data = data.sort_values([var1, 'random'], ascending=True)
        data['targeted_var1'] = targeting_vector
        data = data.sort_values([var2, 'random'], ascending=True)
        data['targeted_var2'] = targeting_vector

        # Get demographic parity and poverty share for each group
        results: Dict[str, Dict[str, float]] = {}
        for group in data[characteristic].unique():
            subset = data[data[characteristic] == group]
            results[group] = {}
            results[group]['poverty_share'] = 100 * subset['targeted_var1'].mean()
            results[group]['demographic_parity'] = 100 * subset['targeted_var2'].mean() - \
                                                   100 * subset['targeted_var1'].mean()
        return results

    def rank_residuals_plot(self, groundtruth: str, proxies: List[str], characteristic: str,
                            weighted: bool = False, colors: Optional[sns.color_palette] = None) -> None:
        """
        Plots the rank-residuals, as box plots, between a list of proxies and the ground-truth data, disaggregated by
        a (demographic) characteristic.

        Args:
            groundtruth: The name of the groundtruth column in the fairness dataset.
            proxies: The list of targeting methods to be compared against each other; each name should have a
                corresponding column in the fairness dataset.
            characteristic: The (demographic) characteristic by which to disaggregate results - this should be the name
                of the corresponding column in the targeting dataset.
            weighted: If True the weighted version of the dataset will be used.
            colors: The color palette to use when plotting.
        """

        data = self.ds.weighted_fairness if weighted else self.ds.unweighted_fairness

        # Set up figure
        fig, ax = plt.subplots(1, len(proxies), figsize=(20, 7), sharey=True)
        max_resid = 0

        # Set up color palette
        if colors is None:
            colors = self.default_colors

        for i, proxy in enumerate(proxies):

            # Obtain rank residual distribution for each group
            distributions: List[Mapping[str, np.ndarray]] = []
            groups: List[int] = []
            results = self.rank_residual(groundtruth, proxy, characteristic, weighted=weighted)
            for g, group in enumerate(sorted(data[characteristic].unique())):
                distribution = list(results[group])
                distributions = distributions + distribution
                groups = groups + [group for _ in range(len(distribution))]

            # Combine rank residual distributions for all groups into dataframe
            results_df = pd.DataFrame([distributions, groups]).T
            results_df.columns = ['Normalized Rank Residual', characteristic]
            results_df = results_df.sort_values(characteristic, ascending=True)

            # Update range for plotting
            if np.abs(results_df['Normalized Rank Residual'].max()) > max_resid:
                max_resid = np.abs(results_df['Normalized Rank Residual'].max())
            if np.abs(results_df['Normalized Rank Residual'].min()) > max_resid:
                max_resid = np.abs(results_df['Normalized Rank Residual'].min())

            # Create boxplot
            sns.boxplot(data=results_df, x='Normalized Rank Residual', y=characteristic, orient='h', ax=ax[i],
                        showfliers=False,
                        palette={group: colors[g] for g, group in enumerate(sorted(data[characteristic].unique()))})
            ax[i].axvline(0, color='grey', dashes=[1, 1])

            # Clean up subplot
            ax[i].set_ylabel('')
            ax[i].set_xlabel('Error')
            ax[i].set_title(proxy, fontsize='large')

        # Clean up overall plot
        for i in range(len(ax)):
            ax[i].set_xlim(-max_resid, max_resid)
            clean_plot(ax[i])
        plt.tight_layout(rect=[0, 0, 1, .92])
        plt.suptitle('Rank Fairness: ' + characteristic, fontsize='x-large')

        # Save and show
        plt.savefig(self.outputs + '/rank_residuals_plot_' + characteristic + '.png', dpi=400)
        plt.show()

    def rank_residuals_table(self, groundtruth: str, proxies: List[str], characteristic: str,
                             weighted: bool = False) -> PandasDataFrame:
        """
        Returns a dataframe with the rank-residuals of all sub-groups part of 'characteristic', and for all proxies
        specified by 'proxies'. It also performs an ANOVA test.

        Args:
            groundtruth: The name of the groundtruth column in the fairness dataset.
            proxies: The list of targeting methods to be compared against each other; each name should have a
                corresponding column in the fairness dataset.
            characteristic: The (demographic) characteristic by which to disaggregate results - this should be the name
                of the corresponding column in the targeting dataset.
            weighted: If True the weighted version of the dataset will be used.

        Returns: The pandas dataframe of rank-residuals and ANOVA F-stat and p-value.
        """

        data = self.ds.weighted_fairness if weighted else self.ds.unweighted_fairness

        # Set up table
        table = pd.DataFrame()
        table[characteristic] = sorted(data[characteristic].unique()) + ['Anova F-Stat', 'Anova p-value']

        for i, proxy in enumerate(proxies):

            # Obtain rank residual distribution for each group
            distributions, means = [], []
            results = self.rank_residual(groundtruth, proxy, characteristic, weighted=weighted)
            for group in sorted(data[characteristic].unique()):
                distribution = list(results[group])
                distributions.append(distribution)
                means.append('%.2f (%.2f)' % (np.mean(distribution), np.std(distribution)))

            # Anova test to determine whether means of distributions are statistically significantly different
            anova = f_oneway(*tuple(distributions))
            column = means + [anova[0]] + [anova[1]]

            # Add column to table
            table[proxy] = column

        # Save and return
        table.to_csv(self.outputs + '/rank_residuals_table_' + characteristic + '.png', index=False)
        return table

    def demographic_parity_table(self, groundtruth: str, proxies: List[str], characteristic: str,
                                 p: Union[float, int], weighted: bool = False, format_table: bool = True
                                 ) -> PandasDataFrame:
        """
        Returns a dataframe with the demographic parity vales of all sub-groups part of 'characteristic', and for all
        proxies specified by 'proxies'.

        Args:
            groundtruth: The name of the groundtruth column in the fairness dataset.
            proxies: The list of targeting methods to be compared against each other; each name should have a
                corresponding column in the fairness dataset.
            characteristic: The (demographic) characteristic by which to disaggregate results - this should be the name
                of the corresponding column in the targeting dataset.
            p: The percentile below which users will be considered as poor.
            weighted: If True the weighted version of the dataset will be used.
            format_table: If true, provide only two significant decimal digits.

        Returns: The pandas dataframe with DP values, each group's share of the population, and its share of the target
        population.
        """

        data = self.ds.weighted_fairness if weighted else self.ds.unweighted_fairness
        data['count'] = 1

        # Set up table
        table = pd.DataFrame()
        groups = sorted(data[characteristic].unique())
        table[characteristic] = groups

        # Get share of population for each group
        population_shares = data.groupby(characteristic).agg('count')
        table["Group's share of population"] = 100 * (population_shares['count'].values.flatten() / len(data))

        # Get demographic parity and poverty share for each group
        for proxy in proxies:
            groups = sorted(data[characteristic].unique())
            results = self.demographic_parity(groundtruth, proxy, characteristic, p, weighted=weighted)
            if proxy == proxies[0]:
                table["Share of Group in Target Population"] = [results[group]['poverty_share'] for group in groups]
            table[proxy] = [results[group]['demographic_parity'] for group in groups]

        # Clean up and return table
        if format_table:
            table["Group's share of population"] = table["Group's share of population"].apply(
                lambda x: ('%.2f' % x) + '%')
            table["Share of Group in Target Population"] = table["Share of Group in Target Population"].apply(
                lambda x: ('%.2f' % x) + '%')

        # Save and return
        table.to_csv(self.outputs + '/demographic_parity_table_' + characteristic + '_' + str(p) + '%.png', index=False)
        return table

    def demographic_parity_plot(self, groundtruth: str, proxies: List[str], characteristic: str,
                                p: Union[float, int], weighted: bool = False) -> None:
        """
        Plots the demographic parity values, with proxies on the x-axis and subgroups on the y-axis. The magnitude of
        the value determines the corresponding circle's radius, while its sign the circle's color (red for positive,
        blue for negative).

        Args:
            groundtruth: The name of the groundtruth column in the fairness dataset.
            proxies: The list of targeting methods to be compared against each other; each name should have a
                corresponding column in the fairness dataset.
            characteristic: The (demographic) characteristic by which to disaggregate results - this should be the name
                of the corresponding column in the targeting dataset.
            p: The percentile below which users will be considered as poor.
            weighted: If True the weighted version of the dataset will be used.
        """

        # Get demographic parity table, set up parameters for grid 
        table = self.demographic_parity_table(groundtruth, proxies, characteristic, p, weighted=weighted,
                                              format_table=False)
        data = table[proxies]
        keys = list(data.keys())
        N = len(table)
        M = len(keys)

        # Format labels for y axis
        ylabels = []
        for i in range(len(table)):
            ylabels.append('{}\n{}% of Population\n{}% in Target'
                           .format(table.iloc[i][characteristic], int(table.iloc[i]["Group's share of population"]),
                                   int(table.iloc[i]['Share of Group in Target Population'])))
        ylabels = ylabels[::-1]
        xlabels = keys

        # Circles
        x, y = np.meshgrid(np.arange(M), np.arange(N))
        radius = 15
        s: List[List[float]] = [[] for _ in range(len(keys))]
        for i in range(len(keys)):
            for j in range(len(data[keys[i]])):
                s[i].append(data[keys[i]][j])
        arr = np.array(s).transpose()
        new_list = []
        for i in range(arr.shape[0] - 1, -1, -1):
            new_list.append(list(arr[i]))
        s = new_list

        # set up figure
        fig = plt.figure(figsize=(2.7 * len(proxies), 2.5 * len(table)))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_title('Demographic Parity: ' + characteristic, pad=85, fontsize='large')

        # More circles
        R = np.array([np.array(row) for row in s])
        c = R
        R = np.log(np.abs(R), where=np.abs(R) > 0) / 10
        circles = [plt.Circle((j, i), radius=r) for r, j, i in zip(R.flatten(), x.flatten(), y.flatten())]
        col = PatchCollection(circles, array=c.flatten(), cmap="RdBu_r", edgecolor='grey', linewidth=2)
        col.set_clim(vmin=-20, vmax=20)
        # math.log(abs(r)) / 10

        # Set up ticks and labels
        ax.add_collection(col)
        ax.set(xticks=np.arange(M), yticks=np.arange(N),
               xticklabels=xlabels, yticklabels=ylabels)
        ax.set_xticks(np.arange(M + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(N + 1) - 0.5, minor=True)
        ax.xaxis.tick_top()

        # Colorbar
        cbar = fig.colorbar(col, fraction=0.03, pad=0.05, )
        cbar.outline.set_edgecolor('white')
        cbar.ax.set_ylabel('Percentage Point Difference', labelpad=20)

        # Final touches on figure
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)

        # Save and show
        plt.savefig(self.outputs + '/demographic_parity_plot_' + characteristic + '_' + str(p) + '%.png')
        plt.show()
