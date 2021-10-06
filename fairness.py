"""
Evaluates fairness of a machine learning module across a characteristic 
(whether or not the machine learning module discriminates across different groups in that characteristic).
"""
from numpy import character
from box import Box
import yaml
from helpers.utils import *
from helpers.plot_utils import *
from helpers.io_utils import *
from helpers.ml_utils import *
from scipy.stats import f_oneway, chi2_contingency
from datastore import *
from sklearn.metrics import recall_score, precision_score


class Fairness:

    def __init__(self, datastore: DataStore, dataframe=None, clean_folders=False):
        self.cfg = datastore.cfg
        self.ds = datastore

        data_path = self.cfg.path.data + self.cfg.path.file_names.fairness
        self.data = pd.read_csv(data_path)
        self.data['random'] = np.random.rand(len(self.data))

        outputs = self.cfg.path.outputs
        self.outputs = outputs
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
        self.weighted_data = pd.DataFrame(np.repeat(self.weighted_data.values, self.weighted_data['weight'], axis=0), columns=self.weighted_data.columns)\
            .astype(self.unweighted_data.dtypes)

    def rank_residual(self, var1, var2, characteristic, weighted=False):
        
        data = self.weighted_data if weighted else self.unweighted_data
        
        # Get rankings according to var1 and var2
        data = data.sort_values([var1, 'random'], ascending=True)
        data['rank1'] = range(len(data))
        data = data.sort_values([var2, 'random'], ascending=True)
        data['rank2'] = range(len(data))

        # Calcualte normalized rank residual
        data['rank_residual'] = (data['rank2'] - data['rank1'])/data['rank2'].max()

        # Return rank residual distributions for each group
        return {group: data[data[characteristic] == group]['rank_residual'].values.flatten() for group in data[characteristic].unique()}

    def demographic_parity(self, var1, var2, characteristic, p, weighted=False):
        """
        var1 = groundtruth
        var2 = proxies. The thing that you're evaluating the fairness of.
        characteristic = how the group is divided
        p = target the top p percent
        """

        data = self.weighted_data if weighted else self.unweighted_data

        # Simulate targeting by var1 and var2
        num_ones = int((p/100)*len(data))
        num_zeros = len(data) - num_ones
        targeting_vector = np.concatenate([np.ones(num_ones), np.zeros(num_zeros)])
        data = data.sort_values([var1, 'random'], ascending=True)
        data['targeted_var1'] = targeting_vector
        data = data.sort_values([var2, 'random'], ascending=True)
        data['targeted_var2'] = targeting_vector

        # Get demographic parity and poverty share for each group
        results = {}
        for group in data[characteristic].unique():
            subset = data[data[characteristic] == group]
            results[group] = {}
            results[group]['poverty_share'] = 100*subset['targeted_var1'].mean()
            results[group]['demographic_parity'] = 100*subset['targeted_var2'].mean() - 100*subset['targeted_var1'].mean()
        return results
    
    def independence(self, var1, var2, characteristic, p, weighted=False):
        """
        Performs the chi-squared independence test between the proxy variable and the sensitive characterisrtic

        var2 -- the proxy variable
        characteristic -- sensitive characteristic

        """
        data = self.weighted_data if weighted else self.unweighted_data

        # Simulate targeting by var2 (proxy characteristic)
        num_ones = int((p/100)*len(data))
        num_zeros = len(data) - num_ones
        targeting_vector = np.concatenate([np.ones(num_ones), np.zeros(num_zeros)])
        data = data.sort_values([var2, 'random'], ascending=True)
        data['targeted_var2'] = targeting_vector

        # Create a pivot table between the characteristic and the proxy
        pivot = data.pivot_table(index=characteristic, columns='targeted_var2', aggfunc='count', fill_value=0).iloc[:, 0:2]

        # Run independence test
        chi2, p, dof, ex = chi2_contingency(pivot)
        return p
    
    def recall_per_group(self, var1, var2, characteristic, p, weighted=False):
        """
        var1 = groundtruth
        var2 = proxies. The thing that you're evaluating the fairness of.
        characteristic = how the group is divided
        p = target the top p percent
        """

        data = self.weighted_data if weighted else self.unweighted_data

        # Simulate targeting by var1 and var2
        num_ones = int((p/100)*len(data))
        num_zeros = len(data) - num_ones
        targeting_vector = np.concatenate([np.ones(num_ones), np.zeros(num_zeros)])
        data = data.sort_values([var1, 'random'], ascending=True)
        data['targeted_var1'] = targeting_vector
        data = data.sort_values([var2, 'random'], ascending=True)
        data['targeted_var2'] = targeting_vector

        # Get demographic parity and poverty share for each group
        results = {}
        for group in data[characteristic].unique():
            subset = data[data[characteristic] == group]
            results[group] = recall_score(subset['targeted_var1'], subset['targeted_var2'])
        return results
    
    def precision_per_group(self, var1, var2, characteristic, p, weighted=False):
        """
        var1 = groundtruth
        var2 = proxies. The thing that you're evaluating the fairness of.
        characteristic = how the group is divided
        p = target the top p percent
        """

        data = self.weighted_data if weighted else self.unweighted_data

        # Simulate targeting by var1 and var2
        num_ones = int((p/100)*len(data))
        num_zeros = len(data) - num_ones
        targeting_vector = np.concatenate([np.ones(num_ones), np.zeros(num_zeros)])
        data = data.sort_values([var1, 'random'], ascending=True)
        data['targeted_var1'] = targeting_vector
        data = data.sort_values([var2, 'random'], ascending=True)
        data['targeted_var2'] = targeting_vector

        # Get demographic parity and poverty share for each group
        results = {}
        for group in data[characteristic].unique():
            subset = data[data[characteristic] == group]
            results[group] = precision_score(subset['targeted_var1'], subset['targeted_var2'])
        return results

    def rank_residuals_plot(self, groundtruth, proxies, characteristic, weighted=False, colors=None):

        data = self.weighted_data if weighted else self.unweighted_data

        # Set up figure
        fig, ax = plt.subplots(1, len(proxies), figsize=(20, 7), sharey=True)
        max_resid = 0

        # Set up color palette
        if colors is None:
            colors = self.default_colors

        for i, proxy in enumerate(proxies):

            # Obtain rank residual distribution for each group
            distributions, groups = [], []
            results = self.rank_residual(groundtruth, proxy, characteristic, weighted=weighted)
            for g, group in enumerate(sorted(data[characteristic].unique())):
                distribution = list(results[group])
                distributions = distributions + distribution
                groups = groups + [group for _ in range(len(distribution))]

            # Combine rank residual distributions for all groups into dataframe
            results = pd.DataFrame([distributions, groups]).T
            results.columns = ['Normalized Rank Residual', characteristic]
            results = results.sort_values(characteristic, ascending=True)

            # Update range for plotting
            if np.abs(results['Normalized Rank Residual'].max()) > max_resid:
                max_resid = np.abs(results['Normalized Rank Residual'].max())
            if np.abs(results['Normalized Rank Residual'].min()) > max_resid:
                max_resid = np.abs(results['Normalized Rank Residual'].min())

            # Create boxplot
            sns.boxplot(data=results, x='Normalized Rank Residual', y=characteristic, orient='h', ax=ax[i], showfliers=False, 
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

    def rank_residuals_table(self, groundtruth, proxies, characteristic, weighted=False):

        data = self.weighted_data if weighted else self.unweighted_data

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

    def demographic_parity_table(self, groundtruth, proxies, characteristic, p, weighted=False, format_table=True):

        data = self.weighted_data if weighted else self.unweighted_data
        data['count'] = 1

        # Set up table
        table=pd.DataFrame()
        groups = sorted(data[characteristic].unique())
        table[characteristic] = groups

        # Get share of population for each group
        population_shares = data.groupby(characteristic).agg('count')
        table["Group's share of population"] = 100*(population_shares['count'].values.flatten()/len(data))

        # Get demographic parity and poverty share for each group
        for proxy in proxies:
            groups = sorted(data[characteristic].unique())
            results = self.demographic_parity(groundtruth, proxy, characteristic, p, weighted=weighted)
            if proxy == proxies[0]:
                table["Share of Group in Target Population"] = [results[group]['poverty_share'] for group in groups] 
            table[proxy] = [results[group]['demographic_parity'] for group in groups]
        
        # Clean up and return table
        if format_table:
            table["Group's share of population"] = table["Group's share of population"].apply(lambda x: ('%.2f' % x) + '%')
            table["Share of Group in Target Population"] = table["Share of Group in Target Population"].apply(lambda x: ('%.2f' % x) + '%')

        # Save and return
        table.to_csv(self.outputs + '/demographic_parity_table_' + characteristic + '_' + str(p) + '%.png', index=False)
        return table

    def demographic_parity_plot(self, groundtruth, proxies, characteristic, p, weighted=False):

        # Get demographic parity table, set up parameters for grid 
        table = self.demographic_parity_table(groundtruth, proxies, characteristic, p, weighted=weighted, format_table=False)
        data = table[proxies]
        keys = list(data.keys())
        N = len(table)
        M = len(keys)

        # Format labels for y axis
        ylabels = []
        for i in range(len(table)):
            ylabels.append('{}\n{}% of Population\n{}% in Target'\
                        .format(table.iloc[i][characteristic], int(table.iloc[i]["Group's share of population"]), 
                        int(table.iloc[i]['Share of Group in Target Population'])))
        ylabels = ylabels[::-1]
        xlabels = keys

        # Circles
        x, y = np.meshgrid(np.arange(M), np.arange(N))
        radius = 15
        s = [[] for i in range(len(keys))]
        for i in range(len(keys)):
            for j in range(len(data[keys[i]])):
                s[i].append(data[keys[i]][j])
        arr = np.array(s).transpose()
        new_list = []
        for i in range(arr.shape[0]-1,-1,-1):
            new_list.append(list(arr[i]))
        s = new_list

        # set up figure
        fig = plt.figure(figsize=(2.7*len(proxies), 2.5*len(table))) 
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_title('Demographic Parity: ' + characteristic,pad=85, fontsize='large')

        # More circles
        s=np.array([np.array(row) for row in s])
        R = s
        c = R
        R = np.log(np.abs(R))/10
        circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.flatten(), x.flatten(), y.flatten())]
        col = PatchCollection(circles, array=c.flatten(), cmap="RdBu_r", edgecolor='grey', linewidth=2)
        col.set_clim(vmin=-20, vmax=20)
        # math.log(abs(r)) / 10

        # Set up ticks and labels
        ax.add_collection(col)
        ax.set(xticks=np.arange(M), yticks=np.arange(N),
            xticklabels=xlabels, yticklabels=ylabels)
        ax.set_xticks(np.arange(M+1)-0.5, minor=True)
        ax.set_yticks(np.arange(N+1)-0.5, minor=True)
        ax.xaxis.tick_top()

        # Colorbar
        cbar = fig.colorbar(col, fraction=0.03, pad=0.05,)
        cbar.outline.set_edgecolor('white')
        cbar.ax.set_ylabel('Percentage Point Difference', labelpad=20)

        # Final touches on figure
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        plt.tight_layout()

        # Save and show
        plt.savefig(self.outputs + '/demographic_parity_plot_' + characteristic + '_' + str(p) + '%.png', index=False)
        plt.show()
    
    def recall_table(self, groundtruth, proxies, characteristic, p, weighted=False, format_table=True):

        data = self.weighted_data if weighted else self.unweighted_data
        data['count'] = 1

        # Set up table
        table=pd.DataFrame()
        groups = sorted(data[characteristic].unique())
        table[characteristic] = groups

        # Get share of population for each group
        population_shares = data.groupby(characteristic).agg('count')
        table["Group's share of population"] = 100*(population_shares['count'].values.flatten()/len(data))

        # Get demographic parity and poverty share for each group
        for proxy in proxies:
            groups = sorted(data[characteristic].unique())
            results = self.recall_per_group(groundtruth, proxy, characteristic, p, weighted=weighted)
            table[proxy] = [results[group] for group in groups]
        
        # Clean up and return table
        if format_table:
            table["Group's share of population"] = table["Group's share of population"].apply(lambda x: ('%.2f' % x) + '%')

        # Save and return
        table.to_csv(self.outputs + '/recall_table_' + characteristic + '_' + str(p) + '%.png', index=False)
        return table

    def recall_plot(self, groundtruth, proxies, characteristic, p, weighted=False):

        # Get recall table, set up parameters for grid 
        table = self.recall_table(groundtruth, proxies, characteristic, p, weighted=weighted, format_table=False)
        data = table[proxies]
        keys = list(data.keys())
        N = len(table)
        M = len(keys)

        # Format labels for y axis
        ylabels = []
        for i in range(len(table)):
            # ylabels.append('{}\n{}% of Population\n{}% in Target'\
            #             .format(table.iloc[i][characteristic], int(table.iloc[i]["Group's share of population"]), 
            #             int(table.iloc[i]['Share of Group in Target Population'])))
            ylabels.append('{}\n{}% of Population'\
                        .format(table.iloc[i][characteristic], int(table.iloc[i]["Group's share of population"])))
        ylabels = ylabels[::-1]
        xlabels = keys

        # Circles
        x, y = np.meshgrid(np.arange(M), np.arange(N))
        radius = 15
        s = [[] for i in range(len(keys))]
        for i in range(len(keys)):
            for j in range(len(data[keys[i]])):
                s[i].append(data[keys[i]][j])
        arr = np.array(s).transpose()
        new_list = []
        for i in range(arr.shape[0]-1,-1,-1):
            new_list.append(list(arr[i]))
        s = new_list
        print(s)
        # S contains the recall values, from bottom to top, from left to rightr

        # set up figure
        fig = plt.figure(figsize=(2.7*len(proxies), 2.5*len(table))) 
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_title('Recall per Group: ' + characteristic,pad=85, fontsize='large')

        # More circles
        s=np.array([np.array(row) for row in s])
        R = s
        c = R
        R = np.log(np.abs(R))/10
        circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.flatten(), x.flatten(), y.flatten())]
        col = PatchCollection(circles, array=c.flatten(), cmap="RdBu_r", edgecolor='grey', linewidth=2)
        col.set_clim(vmin=-20, vmax=20)
        # math.log(abs(r)) / 10

        # Set up ticks and labels
        ax.add_collection(col)
        ax.set(xticks=np.arange(M), yticks=np.arange(N),
            xticklabels=xlabels, yticklabels=ylabels)
        ax.set_xticks(np.arange(M+1)-0.5, minor=True)
        ax.set_yticks(np.arange(N+1)-0.5, minor=True)
        ax.xaxis.tick_top()

        # Colorbar
        cbar = fig.colorbar(col, fraction=0.03, pad=0.05,)
        cbar.outline.set_edgecolor('white')
        cbar.ax.set_ylabel('Percentage Point Difference', labelpad=20)

        # Final touches on figure
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        plt.tight_layout()

        # Save and show
        plt.savefig(self.outputs + '/recall_plot_' + characteristic + '_' + str(p) + '%.png', index=False)
        plt.show()
    
    def precision_table(self, groundtruth, proxies, characteristic, p, weighted=False, format_table=True):

        data = self.weighted_data if weighted else self.unweighted_data
        data['count'] = 1

        # Set up table
        table=pd.DataFrame()
        groups = sorted(data[characteristic].unique())
        table[characteristic] = groups

        # Get share of population for each group
        population_shares = data.groupby(characteristic).agg('count')
        table["Group's share of population"] = 100*(population_shares['count'].values.flatten()/len(data))

        # Get demographic parity and poverty share for each group
        for proxy in proxies:
            groups = sorted(data[characteristic].unique())
            results = self.precision_per_group(groundtruth, proxy, characteristic, p, weighted=weighted)
            table[proxy] = [results[group] for group in groups]
        
        # Clean up and return table
        if format_table:
            table["Group's share of population"] = table["Group's share of population"].apply(lambda x: ('%.2f' % x) + '%')

        # Save and return
        table.to_csv(self.outputs + '/recall_table_' + characteristic + '_' + str(p) + '%.png', index=False)
        return table