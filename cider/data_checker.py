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

from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import dask.dataframe as dd
import numpy as np
import pandas as pd
from pandas import DataFrame as PandasDataFrame

from cider.datastore import DataStore, DataType
from helpers.plot_utils import clean_plot, dates_xaxis
from helpers.utils import make_dir, save_df


class DataChecker:
    
    def __init__(
        self,
        datastore: DataStore,
        clean_folders: bool = False
    ) -> None:

        self.cfg = datastore.cfg
        self.datastore = datastore
        self.data_format = datastore.data_format
                
        self.outputs_path = self.cfg.path.working.directory_path / 'data_checker'
        make_dir(self.outputs_path, clean_folders)
        make_dir(self.outputs_path / "datasets")
        make_dir(self.outputs_path / "plots")


    # TODO: change signature maybe
    def load_data(
        self,
        data_type: DataType,
        dataframe: Optional[Union[PandasDataFrame, dd.DataFrame]] = None
    ):

        data_type_map = {
            data_type: dataframe
        }

        self.datastore.load_data(data_type_map=data_type_map, all_required=True)


    def round_fraction(self, x: Union[float, pd.Series]) -> Union[float, pd.Series]:
        return np.round(x, 2)

    
    def display_dataframe(self, df: Union[PandasDataFrame, dd.DataFrame]):
        
        # TODO: Improve
        ipython = True
        
        if isinstance(df, dd.DataFrame):
            df = df.compute()

        if ipython:
            display(df)
        else:    
            with pd.option_context(
                'display.max_rows', 50,
                'display.max_columns', None,
                'display.width', 1000,
                'display.precision', 3,
                'display.colheader_justify', 'center'
            ):
                print(df)

    def summarize_nulls(self, df_name, column_to_disaggregate: str = None):
        
        df = getattr(self.datastore, df_name)

        if isinstance(df, dd.DataFrame):
            total_count = len(df)
            null_counts = df.isna().sum().compute()
        else:
            total_count = len(df)
            null_counts = df.isna().sum()

        null_fractions = pd.DataFrame([{"total_count": total_count, **null_counts.to_dict()}])
        for c in null_counts.index:
            null_fractions[c] = self.round_fraction(null_fractions[c] / null_fractions["total_count"])
        
        print(f'Nulls by column in {df_name} table\n')
        self.display_dataframe(null_fractions)
        if column_to_disaggregate:
            if isinstance(df, dd.DataFrame):
                pdf = df.compute()
            else:
                pdf = df

            rows = []
            for key, g in pdf.groupby(column_to_disaggregate):
                total = len(g)
                nulls = g.isna().sum().to_dict()
                row = {column_to_disaggregate: key, "total_count": total, **nulls}
                for c in g.columns:
                    if c in (column_to_disaggregate,):
                        continue
                    row[c] = self.round_fraction((nulls.get(c, 0) / total) if total else 0.0)
                rows.append(row)
            null_fractions_disaggregated = pd.DataFrame(rows)

            print(f'Nulls by column in {df_name} table, disaggregated by {column_to_disaggregate}\n')
            self.display_dataframe(null_fractions_disaggregated)


    def summarize_timestamp_column(self, df_name, column_name):
        
        df = getattr(self.datastore, df_name)
        if isinstance(df, dd.DataFrame):
            ts = dd.to_datetime(df[column_name], errors="coerce")
            count_not_null = ts.notna().sum().compute()
            earliest = ts.min().compute()
            latest = ts.max().compute()
            has_time = ((ts.dt.hour.fillna(0) != 0) | (ts.dt.minute.fillna(0) != 0) | (ts.dt.second.fillna(0) != 0))
            num_with_time = has_time.sum().compute()
        else:
            ts = pd.to_datetime(df[column_name], errors="coerce")
            count_not_null = ts.notna().sum()
            earliest = ts.min()
            latest = ts.max()
            has_time = ((ts.dt.hour.fillna(0) != 0) | (ts.dt.minute.fillna(0) != 0) | (ts.dt.second.fillna(0) != 0))
            num_with_time = int(has_time.sum())

        summary = pd.DataFrame([{
            "column_name": column_name,
            "count_not_null": int(count_not_null),
            "earliest": earliest,
            "latest": latest,
            "fraction_with_time_of_day": (num_with_time / count_not_null) if count_not_null else 0.0
        }])
        
        print(f'Temporal summary of {df_name}.{column_name}\n')
        
        self.display_dataframe(summary)


    def summarize_unique_subscribers(self, df_names: List[str], column_names: List[str]):

        assert len(df_names) == len(column_names)

        labels = []
        unique_subscriber_lists = []
        for df_name, column_name in zip(df_names, column_names):
            labels.append(f'{df_name}.{column_name}')
            df = getattr(self.datastore, df_name)
            if isinstance(df, dd.DataFrame):
                unique_subscriber_lists.append(df[[column_name]].rename(columns={column_name: "subscribers"}).drop_duplicates())
            else:
                unique_subscriber_lists.append(df[[column_name]].rename(columns={column_name: "subscribers"}).drop_duplicates())

        if hasattr(self.datastore, 'phone_numbers_to_featurize'):

            phone_numbers_to_featurize = getattr(self.datastore, 'phone_numbers_to_featurize')
            phone_numbers_to_featurize_colname = phone_numbers_to_featurize.columns[0]
            num_phone_numbers_to_featurize = len(phone_numbers_to_featurize)

            numbers_to_featurize_comparison_table = [
                {
                    'name': 'phone_numbers_to_featurize',
                    '# unique subscribers': num_phone_numbers_to_featurize,
                    '# matching nums to featurize': pd.NA,
                    'fraction matching nums to featurize': pd.NA,
                    'fraction of nums to featurize covered': pd.NA
                }
            ]


            for label, unique_subscriber_list in zip(labels, unique_subscriber_lists):

                row = dict()

                total_unique_subscribers = len(unique_subscriber_list)
                if isinstance(unique_subscriber_list, dd.DataFrame) or isinstance(phone_numbers_to_featurize, dd.DataFrame):
                    number_matching = len(
                        unique_subscriber_list.merge(
                            phone_numbers_to_featurize.rename(columns={phone_numbers_to_featurize_colname: "subscribers"}),
                            on="subscribers",
                            how="inner",
                        ).drop_duplicates()
                    )
                else:
                    number_matching = len(
                        unique_subscriber_list.merge(
                            phone_numbers_to_featurize.rename(columns={phone_numbers_to_featurize_colname: "subscribers"}),
                            on="subscribers",
                            how="inner",
                        ).drop_duplicates()
                    )
                row['name']= label
                row['# unique subscribers'] = total_unique_subscribers
                row['# matching nums to featurize'] = number_matching
                row['fraction matching nums to featurize'] = number_matching / total_unique_subscribers
                row['fraction of nums to featurize covered'] = number_matching / num_phone_numbers_to_featurize

                numbers_to_featurize_comparison_table.append(row)

            numbers_to_featurize_comparison_table = pd.DataFrame.from_dict(numbers_to_featurize_comparison_table).set_index('name')
            print('Summary of subscriber counts compared with phone numbers to featurize')
            self.display_dataframe(numbers_to_featurize_comparison_table)

        else:
            print('No list of phone numbers to featurize loaded; skipping checks which require them.')

        print('\n')

        pairwise_table = np.full((len(df_names), len(df_names)), pd.NA)

        for i, unique_subscriber_list_i in enumerate(unique_subscriber_lists):

            for j, unique_subscriber_list_j in enumerate(unique_subscriber_lists):

                if i == j:
                    pairwise_table[i,i] = len(unique_subscriber_list_i)

                elif i > j:
                    pairwise_table[i,j] = len(
                        unique_subscriber_list_i.merge(unique_subscriber_list_j, on="subscribers", how="inner").drop_duplicates()
                    )
        print(
            'Summary of subscriber counts between datasets\n'
            '* Diagonal contains # unique subscribers for each dataset/column\n'
            '* Off-diagonal contains the number of unique subscribers in common between datasets/columns'
        )
        self.display_dataframe(pd.DataFrame(pairwise_table, columns=labels, index=labels))

        
    def check_antenna_coverage_in_cdr(self):
    
        try:
            cdr = self.datastore.cdr
            antennas = self.datastore.antennas

        except AttributeError:

            print("CDR and/or antennas missing; can't check CDR antenna coverage.")
            raise

        # TODO: Remove drop-duplicates once I drop duplicates on antenna load.
        antennas = antennas.drop_duplicates(subset=['antenna_id'])
        transaction_count = len(cdr)

        rows = []
        for antenna_column in ('caller_antenna', 'recipient_antenna'):

            joined = cdr.merge(antennas, left_on=antenna_column, right_on="antenna_id", how="inner")
            count_with_antenna = len(joined)
            if isinstance(joined, dd.DataFrame):
                count_with_lat_lon = len(joined.dropna(subset=["latitude", "longitude"]))
            else:
                count_with_lat_lon = len(joined.dropna(subset=["latitude", "longitude"]))

            row = {
                'column': f'cdr.{antenna_column}',
                '# found in antenna dataset': count_with_antenna,
                'fraction found in antenna dataset': count_with_antenna / transaction_count,
                '# with lat & lon': count_with_lat_lon,
                'fraction with lat&lon': count_with_lat_lon / transaction_count
            }
            rows.append(row)

        table = pd.DataFrame.from_dict(rows).set_index('column')
        
        print('CDR antenna coverage summary')
        print(f'Total CDR transactions: {transaction_count}')

        self.display_dataframe(table.round(2))


    def plot_timeseries(self, df_name):
        
        df = getattr(self.datastore, df_name)
        outputs_path = self.outputs_path

        (outputs_path / 'plots').mkdir(exist_ok=True)

        name_without_spaces = df_name.replace(' ', '').lower()

        if 'txn_type' not in df.columns:
            df["txn_type"] = "txn"

        # Save timeseries of transactions by day
        save_df(df.groupby(['txn_type', 'day']).size().reset_index(name="count"),
            outputs_path / 'datasets' / f'{name_without_spaces}_transactionsbyday.csv')

        # Save timeseries of subscribers by day
        save_df(
            df.groupby(['txn_type', 'day'])['caller_id'].nunique().reset_index(name="count"),
            outputs_path / 'datasets' / f'{name_without_spaces}_subscribersbyday.csv'
        )

        timeseries = pd.read_csv(
            outputs_path / 'datasets' / f'{name_without_spaces}_transactionsbyday.csv'
        )

        timeseries['day'] = pd.to_datetime(timeseries['day'])
        timeseries = timeseries.sort_values('day', ascending=True)
        fig, ax = plt.subplots(1, figsize=(20, 6))
        for txn_type in timeseries['txn_type'].unique():
            subset = timeseries[timeseries['txn_type'] == txn_type]
            ax.plot(subset['day'], subset['count'], label=txn_type)
            ax.scatter(subset['day'], subset['count'], label='')
        if len(timeseries['txn_type'].unique()) > 1:
            ax.legend(loc='best')
        ax.set_title(df_name + ' Transactions by Day', fontsize='large')
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)
        dates_xaxis(ax, frequency='week')
        clean_plot(ax)
        plt.savefig(outputs_path / 'plots' / f'{name_without_spaces}_transactionsbyday.png', dpi=300)

        # Plot timeseries of subscribers by day
        timeseries = pd.read_csv(
            outputs_path /'datasets' / f'{name_without_spaces}_subscribersbyday.csv'
        )
        timeseries['day'] = pd.to_datetime(timeseries['day'])
        timeseries = timeseries.sort_values('day', ascending=True)
        fig, ax = plt.subplots(1, figsize=(20, 6))
        for txn_type in timeseries['txn_type'].unique():
            subset = timeseries[timeseries['txn_type'] == txn_type]
            ax.plot(subset['day'], subset['count'], label=txn_type)
            ax.scatter(subset['day'], subset['count'], label='')
        if len(timeseries['txn_type'].unique()) > 1:
            ax.legend(loc='best')
        ax.set_title(df_name + ' Subscribers by Day', fontsize='large')
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)
        dates_xaxis(ax, frequency='week')
        clean_plot(ax)
        plt.savefig(outputs_path / 'plots' / f'{name_without_spaces}_subscribersbyday.png', dpi=300)
              
        plt.show()