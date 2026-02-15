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

from enum import Enum
import json
import os
from collections import defaultdict
from multiprocessing import Pool
from typing import Any, Dict, Optional, Union

import bandicoot as bc  # type: ignore[import]
import dask.dataframe as dd
import geopandas as gpd  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import]

from helpers.features import all_dask
from helpers.plot_utils import clean_plot, dates_xaxis, distributions_plot
from helpers.utils import (cdr_bandicoot_format, filter_by_phone_numbers_to_featurize,
                           flatten_folder, flatten_lst, get_dask_client,
                           long_join_pandas, long_join_dask, make_dir,
                           read_csv, read_parquet, save_parquet, save_df)
from numpy import nan
from pandas import DataFrame as PandasDataFrame

from .datastore import DataStore, DataType

# Type alias
DaskDataFrame = dd.DataFrame


class _OutputFormat(Enum):
    CSV = 1
    PARQUET = 2


class Featurizer:

    def __init__(self,
                 datastore: DataStore,
                 dataframes: Optional[Dict[str, Optional[Union[PandasDataFrame, DaskDataFrame]]]] = None,
                 clean_folders: bool = False) -> None:

        self.cfg = datastore.cfg
        self.ds = datastore
        outputs_path = self.cfg.path.working.directory_path / 'featurizer'
        self.outputs_path = outputs_path
        # Prepare working directories
        make_dir(outputs_path, clean_folders)
        make_dir(outputs_path / 'outputs')
        make_dir(outputs_path / 'plots')
        make_dir(outputs_path / 'tables')

        self.features: Dict[str, Optional[DaskDataFrame]] = {'cdr': None, 'international': None, 'recharges': None,
                                                              'location': None, 'mobiledata': None, 'mobilemoney': None}

        # Dask setup
        client = get_dask_client(self.cfg)
        self.client = client

        # Create default dicts to avoid key errors
        dataframes = dataframes if dataframes else defaultdict(lambda: None)
        data_type_map = {
            DataType.CDR: dataframes['cdr'],
            DataType.RECHARGES: dataframes['recharges'],
            DataType.MOBILEDATA: dataframes['mobiledata'],
            DataType.MOBILEMONEY: dataframes['mobilemoney'],
            DataType.ANTENNAS: dataframes['antennas'],
            DataType.SHAPEFILES: None,
            DataType.PHONE_NUMBERS_TO_FEATURIZE: None
        }
        # Load data into datastore, initialize bandicoot attribute
        self.ds.load_data(data_type_map=data_type_map, all_required=False)
        self.ds.cdr_bandicoot = None

        self.phone_numbers_to_featurize = getattr(self.ds, 'phone_numbers_to_featurize', None)
        
        if 'params' in self.cfg and 'feature_output_format' in self.cfg.params:

            output_format_string = self.cfg.params.feature_output_format
            if output_format_string == 'parquet':
                self.output_format = _OutputFormat.PARQUET
            elif output_format_string == 'csv':
                self.output_format = _OutputFormat.CSV
            else:
                raise ValueError(f'Unknown feature output format {output_format_string}.')
        else:
            self.output_format = _OutputFormat.CSV


    def diagnostic_statistics(self, write: bool = True) -> Dict[str, Dict[str, int]]:
        """
        Compute summary statistics of datasets

        Args:
            write: whether to write json to disk

        Returns: dict of dicts containing summary stats - {'CDR': {'Transactions': 2.3, ...}, ...}
        """
        statistics: Dict[str, Dict[str, int]] = {}

        for name, df in [('CDR', self.ds.cdr),
                         ('Recharges', self.ds.recharges),
                         ('Mobile Data', self.ds.mobiledata),
                         ('Mobile Money', self.ds.mobilemoney)]:
            if df is not None:

                statistics[name] = {}

                # Number of days
                lastday = pd.to_datetime(df['timestamp'].max().compute())
                firstday = pd.to_datetime(df['timestamp'].min().compute())
                statistics[name]['Days'] = (lastday - firstday).days + 1

                # Number of transactions
                statistics[name]['Transactions'] = len(df)

                # Number of subscribers
                statistics[name]['Subscribers'] = df['caller_id'].nunique().compute()

                # Number of recipients
                if 'recipient_id' in df.columns:
                    statistics[name]['Recipients'] = df['recipient_id'].nunique().compute()

        if write:
            with open(self.outputs_path / 'tables' / 'statistics.json', 'w') as f:
                json.dump(statistics, f)

        return statistics

    def diagnostic_plots(self, plot: bool = True) -> None:
        """
        Compute time series of transactions, save to disk, and plot if requested

        Args:
            plot: whether to plot graphs
        """

        for name, df in [('CDR', getattr(self.ds, 'cdr', None)),
                         ('Recharges', getattr(self.ds, 'recharges', None)),
                         ('Mobile Data', getattr(self.ds, 'mobiledata', None)),
                         ('Mobile Money', getattr(self.ds, 'mobilemoney', None))]:
            if df is not None:
                
                name_without_spaces = name.replace(' ', '')

                if 'txn_type' not in df.columns:
                    df['txn_type'] = 'txn'

                # Save timeseries of transactions by day
                transactions_by_day = df.groupby(['txn_type', 'day']).size().reset_index(name='count')
                save_df(transactions_by_day,
                        self.outputs_path / 'datasets' / f'{name_without_spaces}_transactionsbyday.csv')

                # Save timeseries of subscribers by day
                subscribers_by_day = df.groupby(['txn_type', 'day'])['caller_id'].nunique().reset_index(name='count')
                save_df(subscribers_by_day,
                        self.outputs_path / 'datasets' / f'{name_without_spaces}_subscribersbyday.csv')

                if plot:
                    # Plot timeseries of transactions by day
                    timeseries = pd.read_csv(
                        self.outputs_path / 'datasets' / f'{name_without_spaces}_transactionsbyday.csv'
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
                    ax.set_title(name + ' Transactions by Day', fontsize='large')
                    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)
                    dates_xaxis(ax, frequency='week')
                    clean_plot(ax)
                    plt.savefig(self.outputs_path / 'plots' / f'{name_without_spaces}_transactionsbyday.png', dpi=300)

                    # Plot timeseries of subscribers by day
                    timeseries = pd.read_csv(
                        self.outputs_path /'datasets' / f'{name_without_spaces}_subscribersbyday.csv'
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
                    ax.set_title(name + ' Subscribers by Day', fontsize='large')
                    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)
                    dates_xaxis(ax, frequency='week')
                    clean_plot(ax)
                    plt.savefig(self.outputs_path / 'plots' / f'{name_without_spaces}_subscribersbyday.png', dpi=300)


    def cdr_features(self, bc_chunksize: int = 500000, bc_processes: int = 55) -> None:
        """
        Compute CDR features using bandicoot library and save to disk

        Args:
            bc_chunksize: number of users per chunk
            bc_processes: number of processes to run in parallel
        """
        # Check that CDR is present to calculate international features
        if self.ds.cdr is None:
            raise ValueError('CDR file must be loaded to calculate CDR features.')
        print('Calculating CDR features...')

        # Convert CDR into bandicoot format
        self.ds.cdr_bandicoot = cdr_bandicoot_format(self.ds.cdr, self.ds.antennas, self.cfg.col_names.cdr)

        # Get list of unique subscribers, write to file
        unique_subscribers = self.ds.cdr_bandicoot[['name']].drop_duplicates()
        save_df(unique_subscribers, self.outputs_path / 'datasets' / 'subscribers.csv')
        subscribers = unique_subscribers['name'].compute().tolist()

        subscribers = filter_by_phone_numbers_to_featurize(self.phone_numbers_to_featurize, subscribers, 'name')

        # Make adjustments to chunk size and parallelization if necessary
        if bc_chunksize > len(subscribers):
            bc_chunksize = len(subscribers)
        if bc_processes > int(len(subscribers) / bc_chunksize):
            bc_processes = int(len(subscribers) / bc_chunksize)

        # Make output folders
        make_dir(self.outputs_path / 'datasets' / 'bandicoot_records')
        make_dir(self.outputs_path / 'datasets' / 'bandicoot_features')

        # Get bandicoot features in chunks
        start = 0
        end = 0
        while end < len(subscribers):

            # Get start and end point of chunk
            end = start + bc_chunksize
            chunk = subscribers[start:end]

            # Name outfolders
            recs_folder = self.outputs_path / 'datasets' / 'bandicoot_records' / f'{start}to{end}'
            bc_folder = self.outputs_path / 'datasets' / 'bandicoot_features' / f'{start}to{end}'

            make_dir(bc_folder)

            # Get records for this chunk and write out to csv files per person
            chunk_df = pd.DataFrame({'name': chunk})
            chunk_dask = dd.from_pandas(chunk_df, npartitions=4)
            matched_chunk = self.ds.cdr_bandicoot.merge(chunk_dask, on='name', how='inner')
            
            # Write partitioned by name for bandicoot
            matched_chunk_pd = matched_chunk.compute()
            for name_val in matched_chunk_pd['name'].unique():
                person_data = matched_chunk_pd[matched_chunk_pd['name'] == name_val]
                person_folder = recs_folder / f'name={name_val}'
                person_folder.mkdir(parents=True, exist_ok=True)
                person_data.to_csv(person_folder / 'data.csv', index=False, header=True)

            # Move csv files around on disk to get into position for bandicoot
            n = int(len(chunk) / bc_processes) if bc_processes > 0 else len(chunk)
            subchunks = [chunk[i:i + n] for i in range(0, len(chunk), n)]
            pool = Pool(bc_processes)
            unmatched = pool.map(flatten_folder, [(subchunk, recs_folder) for subchunk in subchunks])
            unmatched = flatten_lst(unmatched)
            pool.close()
            if len(unmatched) > 0:
                print('Warning: lost %i subscribers in file shuffling' % len(unmatched))

            # Calculate bandicoot features
            def get_bc(sub: Any) -> Any:
                return bc.utils.all(bc.read_csv(str(sub), recs_folder, describe=True), summary='extended',
                                    split_week=True,
                                    split_day=True, groupby=None)

            # Write out bandicoot feature files
            def write_bc_chunk(chunk_subscribers):
                results = []
                for sub in chunk_subscribers:
                    if os.path.isfile(recs_folder / f'{sub}.csv'):
                        results.append(get_bc(sub))
                return results

            # Process in chunks
            all_features = []
            for subchunk in subchunks:
                features = write_bc_chunk(subchunk)
                all_features.extend(features)
            
            # Write bandicoot features
            for idx, feature_chunk in enumerate(np.array_split(all_features, bc_processes)):
                bc.to_csv(list(feature_chunk), bc_folder / f'{idx}.csv')
            
            start = start + bc_chunksize

        # Combine all bandicoot features into a single file, fix column names, and write to disk
        cdr_features = read_csv(self.client, self.outputs_path / 'datasets' / 'bandicoot_features' / '*' / '*', header=True)
        cdr_features = cdr_features[[col for col in cdr_features.columns if
                                            ('reporting' not in col) or (col == 'reporting__number_of_records')]]
        cdr_features = cdr_features.rename(columns={c: c if c == 'name' else 'cdr_' + c for c in cdr_features.columns})
        
        if self.output_format == _OutputFormat.CSV:
            save_df(cdr_features, self.outputs_path / 'datasets' / 'bandicoot_features' / 'all', single_file=False)
        else:
            save_parquet(cdr_features, self.outputs_path / 'datasets' / 'bandicoot_features' / 'all')
        self.features['cdr'] = cdr_features


    def cdr_features_dask(self) -> None:
        """
        Compute CDR features using dask and save to disk
        """
        # Check that CDR is present to calculate international features
        if self.ds.cdr is None:
            raise ValueError('CDR file must be loaded to calculate CDR features.')
        print('Calculating CDR features...')

        cdr_features = all_dask(
            self.ds.cdr,
            self.ds.antennas,
            cfg=self.cfg.params.cdr,
            phone_numbers_to_featurize=self.phone_numbers_to_featurize
        )
        cdr_features_df = long_join_dask(cdr_features, on='caller_id', how='outer')
        cdr_features_df = cdr_features_df.rename(columns={'caller_id': 'name'})

        if self.output_format == _OutputFormat.CSV:
            save_df(cdr_features_df, self.outputs_path / 'datasets' / 'cdr_features_dask' / 'all', single_file=False)
        else:
            save_parquet(cdr_features_df, self.outputs_path / 'datasets' / 'cdr_features_dask' / 'all')
        self.features['cdr'] = cdr_features_df

    # Backward compatibility
    cdr_features_spark = cdr_features_dask

    def international_features(self) -> None:
        # Check that CDR is present to calculate international features
        if self.ds.cdr is None:
            raise ValueError('CDR file must be loaded to calculate international features.')
        print('Calculating international features...')

        # Write international transactions to file
        international_trans = self.ds.cdr[self.ds.cdr['international'] == 'international']
        save_df(international_trans, self.outputs_path / 'datasets' / 'international_transactions.csv')

        # Read international calls
        inter = pd.read_csv(self.outputs_path / 'datasets' / 'international_transactions.csv', dtype={'caller_id': 'str'})

        # Calculate list of aggregations by subscriber
        inter_voice = inter[inter['txn_type'] == 'call']
        inter_sms = inter[inter['txn_type'] == 'text']
        lst = [
            ('recipient_id', ['count', 'nunique']),
            ('day', ['nunique']),
            ('duration', ['sum'])
        ]
        feats = []
        for c, agg in lst:
            for subset, name in [(inter, 'all'), (inter_voice, 'call'), (inter_sms, 'text')]:
                grouped = subset[['caller_id', c]].groupby('caller_id', as_index=False).agg(agg)
                grouped.columns = [name + '__' + c + '__' + ag for ag in agg]
                feats.append(grouped)

        # Combine all aggregations together, write to file
        feats_df = long_join_pandas(feats, on='caller_id', how='outer').rename({'caller_id': 'name'}, axis=1)
        feats_df['name'] = feats_df.index

        feats_df = filter_by_phone_numbers_to_featurize(self.phone_numbers_to_featurize, feats_df, 'name')

        feats_df.columns = [c if c == 'name' else 'international_' + c for c in feats_df.columns]

        if self.output_format == _OutputFormat.CSV:
            save_df(feats_df, self.outputs_path / 'datasets' / 'international_feats', single_file=False)
            self.features['international'] = dd.from_pandas(feats_df, npartitions=4)
        else:
            save_parquet(feats_df, self.outputs_path / 'datasets' / 'international_feats')
            self.features['international'] = read_parquet(self.client, self.outputs_path / 'datasets' / 'international_feats')
        

    def location_features(self) -> None:

        # Check that antennas and CDR are present to calculate spatial features
        if self.ds.cdr is None:
            raise ValueError('CDR file must be loaded to calculate spatial features.')
        if self.ds.antennas is None:
            raise ValueError('Antenna file must be loaded to calculate spatial features.')
        print('Calculating spatial features...')

        # If CDR is not available in bandicoot format, calculate it
        if self.ds.cdr_bandicoot is None:
            self.ds.cdr_bandicoot = cdr_bandicoot_format(self.ds.cdr, self.ds.antennas, self.cfg.col_names.cdr)

        # Get dataframe of antennas located within regions
        antennas = self.ds.antennas.compute()
        antennas = gpd.GeoDataFrame(antennas, geometry=gpd.points_from_xy(antennas['longitude'], antennas['latitude']))
        antennas.crs = {"init": "epsg:4326"}
        antennas = antennas[antennas.is_valid]
        for shapefile_name in self.ds.shapefiles.keys():
            shapefile = self.ds.shapefiles[shapefile_name].rename({'region': shapefile_name}, axis=1)
            antennas = gpd.sjoin(antennas, shapefile, predicate='within', how='left').drop('index_right', axis=1)
            antennas[shapefile_name] = antennas[shapefile_name].fillna('Unknown')
        antennas = dd.from_pandas(antennas.drop(['geometry', 'latitude', 'longitude'], axis=1).fillna(''), npartitions=4)

        cdr_bandicoot_filtered = filter_by_phone_numbers_to_featurize(
            self.phone_numbers_to_featurize, self.ds.cdr_bandicoot, 'name'
        )

        # Merge CDR to antennas
        cdr = cdr_bandicoot_filtered.merge(antennas, on='antenna_id', how='left')
        cdr = cdr.fillna({shapefile_name: 'Unknown' for shapefile_name in self.ds.shapefiles.keys()})

        # Get counts by region
        for shapefile_name in self.ds.shapefiles.keys():
            countbyregion = cdr.groupby(['name', shapefile_name]).size().reset_index(name='count')
            save_df(countbyregion, self.outputs_path / 'datasets' / f'countby{shapefile_name}.csv')

        # Get unique regions (and unique towers)
        unique_regions = cdr[['name']].drop_duplicates()
        for shapefile_name in self.ds.shapefiles.keys():
            region_counts = cdr.groupby('name')[shapefile_name].nunique().reset_index(name=f'count(DISTINCT {shapefile_name})')
            unique_regions = unique_regions.merge(region_counts, on='name', how='left')
        if 'tower_id' in cdr.columns:
            tower_counts = cdr.groupby('name')['tower_id'].nunique().reset_index(name='count(DISTINCT tower_id)')
            unique_regions = unique_regions.merge(tower_counts, on='name', how='left')
        save_df(unique_regions, self.outputs_path / 'datasets' / 'uniqueregions.csv')
        feats = pd.read_csv(self.outputs_path / 'datasets' / 'uniqueregions.csv', dtype={'name': 'str'})


        if len(self.ds.shapefiles) > 0:

            count_by_region_compiled = []
            
            # Pivot counts by region
            for shapefile_name in self.ds.shapefiles.keys():
                count_by_region = pd.read_csv(
                    self.outputs_path / 'datasets' / f'countby{shapefile_name}.csv',
                    dtype={'name': 'str'}
                ).pivot(index='name', columns=shapefile_name, values='count').fillna(0)

                count_by_region['total'] = count_by_region.sum(axis=1)

                # Concatenate the existing dataframe with a list of columns containing percent by region info
                count_by_region_to_concat = [count_by_region]
                for c in set(count_by_region.columns) - {'total', 'name'}:
                    count_by_region_percentage = count_by_region[c] / count_by_region['total']
                    count_by_region_percentage.name = c + '_percent'
                    count_by_region_to_concat.append(count_by_region_percentage)

                count_by_region = pd.concat(count_by_region_to_concat, axis=1)

                count_by_region = count_by_region.rename(
                    {region: shapefile_name + '_' + region for region in count_by_region.columns}, axis=1)

                count_by_region_compiled.append(count_by_region)

            count_by_region = long_join_pandas(count_by_region_compiled, on='name', how='outer')
            count_by_region = count_by_region.drop([c for c in count_by_region.columns if 'total' in c], axis=1)

            # Merge counts and unique counts together, write to file
            feats = count_by_region.merge(feats, on='name', how='outer')

        feats.columns = [c if c == 'name' else 'location_' + c for c in feats.columns]
        if self.output_format == _OutputFormat.CSV:
            save_df(feats, self.outputs_path / 'datasets' / 'location_features', single_file=False)
            self.features['location'] = dd.from_pandas(feats, npartitions=4)
        else:
            save_parquet(feats, self.outputs_path / 'datasets' / 'location_features')
            self.features['location'] = read_parquet(self.client, self.outputs_path / 'datasets' / 'location_features')


    def mobiledata_features(self) -> None:

        # Check that mobile internet data is loaded
        if self.ds.mobiledata is None:
            raise ValueError('Mobile data file must be loaded to calculate mobile data features.')
        print('Calculating mobile data features...')
        
        # Aggregate mobiledata use by day, to control for different definitions of "one transaction."
        mobiledata_aggregated_by_day = self.ds.mobiledata.groupby(['caller_id', 'day'])['volume'].sum().reset_index()
        
        mobiledata_aggregated_by_day = filter_by_phone_numbers_to_featurize(
            self.phone_numbers_to_featurize, mobiledata_aggregated_by_day, 'caller_id'
        )

        # Aggregate to obtain features
        feats = mobiledata_aggregated_by_day.groupby('caller_id').agg({
            'volume': ['sum', 'mean', 'min', 'max', 'std'],
            'day': 'nunique'
        }).reset_index()
        
        # Flatten column names
        feats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in feats.columns.values]
        feats = feats.rename(columns={
            'volume_sum': 'total_volume',
            'volume_mean': 'mean_daily_volume',
            'volume_min': 'min_daily_volume',
            'volume_max': 'max_daily_volume',
            'volume_std': 'std_daily_volume',
            'day_nunique': 'num_days',
            'caller_id': 'name'
        })

        feats = feats.rename(columns={c: c if c == 'name' else 'mobiledata_' + c for c in feats.columns})

        self.features['mobiledata'] = feats
        if self.output_format == _OutputFormat.CSV:
            save_df(feats, self.outputs_path / 'datasets' / 'mobiledata_features', single_file=False)
        else:
            save_parquet(feats, self.outputs_path / 'datasets' / 'mobiledata_features')

    def mobilemoney_features(self) -> None:

        # Check that mobile money is loaded
        if self.ds.mobilemoney is None:
            raise ValueError('Mobile money file must be loaded to calculate mobile money features.')
        print('Calculating mobile money features...')
        
        # create dummy columns for missing field (resulting features will all take value N/A, so be
        #  ignored during model-fitting). This logic should eventually be refined.
        for col_name in [
            'sender_balance_before', 'sender_balance_after', 'recipient_balance_before', 'recipient_balance_after'
        ]:
            if col_name not in self.ds.mobilemoney.columns:
                self.ds.mobilemoney[col_name] = np.nan

        # Get outgoing transactions
        sender_cols = ['txn_type', 'caller_id', 'recipient_id', 'day', 'amount', 'sender_balance_before',
                       'sender_balance_after']
        outgoing = self.ds.mobilemoney[sender_cols].rename(columns={
            'caller_id': 'name',
            'recipient_id': 'correspondent_id',
            'sender_balance_before': 'balance_before',
            'sender_balance_after': 'balance_after'
        })
        outgoing['direction'] = 'out'

        # Get incoming transactions
        recipient_cols = ['txn_type', 'caller_id', 'recipient_id', 'day', 'amount', 'recipient_balance_before',
                          'recipient_balance_after']
        incoming = self.ds.mobilemoney[recipient_cols].rename(columns={
            'recipient_id': 'name',
            'caller_id': 'correspondent_id',
            'recipient_balance_before': 'balance_before',
            'recipient_balance_after': 'balance_after'
        })
        incoming['direction'] = 'in'

        # Combine incoming and outgoing with unified schema
        mm = dd.concat([outgoing, incoming], axis=0)
        save_parquet(mm, self.outputs_path / 'datasets' / 'mobilemoney')
        mm = read_parquet(self.client, self.outputs_path / 'datasets' / 'mobilemoney')
        outgoing = mm[mm['direction'] == 'out']
        incoming = mm[mm['direction'] == 'in']

        # Get mobile money features
        features = []
        for dfname, df in [('all', mm), ('incoming', incoming), ('outgoing', outgoing)]:
            # Create dataframe with 'all' txn type by duplicating rows
            df_all = df.copy()
            df_all['txn_type'] = 'all'
            df = dd.concat([df, df_all], axis=0)

            # Compute aggregations using pandas-style syntax
            aggs_pd = df.compute().groupby(['name', 'txn_type']).agg({
                'amount': ['mean', 'min', 'max'],
                'balance_before': ['mean', 'min', 'max'],
                'balance_after': ['mean', 'min', 'max'],
                'correspondent_id': ['count', 'nunique']
            })
            
            # Flatten and pivot
            aggs_pd.columns = ['_'.join(col).strip() for col in aggs_pd.columns.values]
            aggs_pd = aggs_pd.reset_index()
            aggs_pd = aggs_pd.pivot(index='name', columns='txn_type')
            aggs_pd.columns = ['_'.join(col).strip('_') for col in aggs_pd.columns.values]
            aggs_pd = aggs_pd.reset_index()
            
            # Rename columns with amount/balance/txns/contacts
            rename_dict = {}
            for col in aggs_pd.columns:
                if col != 'name':
                    # Parse column name and add df name prefix
                    rename_dict[col] = dfname + '_' + col
            aggs_pd = aggs_pd.rename(columns=rename_dict)
            
            # Convert back to Dask
            aggs = dd.from_pandas(aggs_pd, npartitions=4)
            features.append(aggs)

        # Combine all mobile money features together and save them
        feats = long_join_dask(features, on='name', how='outer')
        feats = feats.rename(columns={c: c if c == 'name' else 'mobilemoney_' + c for c in feats.columns})

        feats = filter_by_phone_numbers_to_featurize(self.phone_numbers_to_featurize, feats, 'name')
        if self.output_format == _OutputFormat.CSV:
            save_df(feats, self.outputs_path / 'datasets' / 'mobilemoney_feats', single_file=False)
        else:
            save_parquet(feats, self.outputs_path / 'datasets' / 'mobilemoney_feats')
        self.features['mobilemoney'] = feats

    def recharges_features(self) -> None:

        if self.ds.recharges is None:
            raise ValueError('Recharges file must be loaded to calculate recharges features.')
        print('Calculating recharges features...')

        feats = self.ds.recharges.groupby('caller_id').agg({
            'amount': ['sum', 'mean', 'min', 'max', 'count'],
            'day': 'nunique'
        }).reset_index()
        
        # Flatten column names
        feats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in feats.columns.values]
        feats = feats.rename(columns={
            'amount_sum': 'sum',
            'amount_mean': 'mean',
            'amount_min': 'min',
            'amount_max': 'max',
            'amount_count': 'count',
            'day_nunique': 'days',
            'caller_id': 'name'
        })

        feats = feats.rename(columns={c: c if c == 'name' else 'recharges_' + c for c in feats.columns})

        feats = filter_by_phone_numbers_to_featurize(self.phone_numbers_to_featurize, feats, 'name')
        if self.output_format == _OutputFormat.CSV:
            save_df(feats, self.outputs_path / 'datasets' / 'recharges_feats', single_file=False)
        else:
            save_parquet(feats, self.outputs_path / 'datasets' / 'recharges_feats')
        self.features['recharges'] = feats

    def load_features(self) -> None:
        """
        Load features from disk if already computed
        """
        data_path = self.outputs_path / 'datasets'

        features = ['cdr', 'cdr', 'international', 'location', 'mobiledata', 'mobilemoney', 'recharges']
        paths_to_datasets = ['bandicoot_features/all', 'cdr_features_dask/all', 'international_feats', 'location_features',
                    'mobiledata_features', 'mobilemoney_feats', 'recharges_feats']

        # Read data from disk if requested
        for feature, path_to_dataset in zip(features, paths_to_datasets):
            if not self.features[feature]:
                full_path = data_path / path_to_dataset
                try:
                    example_file = next(full_path.iterdir())

                except StopIteration:
                    raise ValueError(f'Directory {path_to_dataset} for {feature} features is empty. Error during featurization?')

                except FileNotFoundError:
                    print(f"Could not locate or read data for '{path_to_dataset}'")
                    continue

                if example_file.suffix == '.csv':
                    # Use Dask to read with proper dtypes
                    self.features[feature] = dd.read_csv(
                        str(data_path / f'{path_to_dataset}.csv'),
                        dtype={'caller_id': 'str', 'name': 'str'}
                    )

                elif example_file.suffix == '.parquet':
                    self.features[feature] = read_parquet(self.client, data_path / path_to_dataset)

    def all_features(self, read_from_disk: bool = False) -> None:
        """
        Join all feature datasets together, save to disk, and assign to attribute

        Args:
            read_from_disk: whether to load features from disk
        """
        if read_from_disk:
            self.load_features()

        # Recompute set of all features if it was already computed
        if 'all' in self.features.keys():
            del self.features['all']

        all_features_list = [self.features[key] for key in self.features.keys() if self.features[key] is not None]
        if all_features_list:
            all_features = long_join_dask(all_features_list, how='outer', on='name')

            if self.output_format == _OutputFormat.CSV:
                save_df(all_features, self.outputs_path / 'datasets' / 'features', single_file=False)
            else:
                save_parquet(all_features, self.outputs_path / 'datasets' / 'features')
            self.features['all'] = all_features
        else:
            print('No features have been computed yet.')

    def feature_plots(self, read_from_disk: bool = False) -> None:
        """
        Plot the distribution of a select number of features

        Args:
            read_from_disk: whether to load features from disk
        """
        
        def check_feature_existence(keys, names, feature_dataframe):
            existing_keys = []
            existing_names = []
            for key, name in zip(keys, names):
                if key in feature_dataframe.columns:
                    existing_keys.append(key)
                    existing_names.append(name)
                    
            return existing_keys, existing_names

        if read_from_disk:
            self.load_features()

        # Plot of distributions of CDR features
        if self.features['cdr'] is not None:
            
            # Features are given different names by dask and bandicoot featurization logic; here we check
            # which exists.
            active_days_key = (
                'cdr_active_days__allweek__day__callandtext' 
                if 'cdr_active_days__allweek__day__callandtext' in self.features['cdr'].columns
                else 'active_days_allweek_allday'
            )
            mean_call_duration_key = (
                'cdr_call_duration__allweek__allday__call__mean' 
                if 'cdr_call_duration__allweek__allday__call__mean' in self.features['cdr'].columns
                else 'call_duration_allweek_allday_call_mean'
            )
            number_of_antennas_key = (
                'cdr_number_of_antennas__allweek__allday' 
                if 'cdr_number_of_antennas__allweek__allday' in self.features['cdr'].columns
                else 'number_of_antennas_allweek_allday'
            )
            feature_keys = [active_days_key, mean_call_duration_key, number_of_antennas_key]
            names = ['Active Days', 'Mean Call Duration', 'Number of Antennas']
            feature_keys, names = check_feature_existence(feature_keys, names, self.features['cdr'])
            distributions_plot(self.features['cdr'], feature_keys, names, color='indianred')
            plt.savefig(self.outputs_path / 'plots' / 'cdr.png', dpi=300)
            plt.show()

        # Plot of distributions of international features
        if self.features['international'] is not None:
            feature_keys = ['international_all__recipient_id__count', 'international_all__recipient_id__nunique',
                        'international_call__duration__sum']
            names = ['International Transactions', 'International Contacts', 'Total International Call Time']
            feature_keys, names = check_feature_existence(feature_keys, names, self.features['international'])

            distributions_plot(self.features['international'], feature_keys, names, color='darkorange')
            plt.savefig(self.outputs_path / 'plots' / 'internation.png', dpi=300)
            plt.show()

        # Plot of distributions of recharges features
        if self.features['recharges'] is not None:
            feature_keys = ['recharges_mean', 'recharges_count', 'recharges_days']
            names = ['Mean Recharge Amount', 'Number of Recharges', 'Number of Days with Recharges']
            feature_keys, names = check_feature_existence(feature_keys, names, self.features['recharges'])

            distributions_plot(self.features['recharges'], feature_keys, names, color='mediumseagreen')
            plt.savefig(self.outputs_path / 'plots' / 'recharges.png', dpi=300)
            plt.show()

        # Plot of distributions of mobile data features
        if self.features['mobiledata'] is not None:
            feature_keys = ['mobiledata_total_volume', 'mobiledata_mean_volume', 'mobiledata_num_days']
            names = ['Total Volume (MB)', 'Mean Transaction Volume (MB)', 'Number of Days with Data Usage']
            feature_keys, names = check_feature_existence(feature_keys, names, self.features['mobiledata'])
            distributions_plot(self.features['mobiledata'], feature_keys, names, color='dodgerblue')
            plt.savefig(self.outputs_path / 'plots' / 'mobiledata.png', dpi=300)
            plt.show()

        # Plot of distributions of mobile money features
        if self.features['mobilemoney'] is not None:
            feature_keys = ['mobilemoney_all_all_amount_mean', 'mobilemoney_all_all_balance_before_mean',
                        'mobilemoney_all_all_txns', 'mobilemoney_all_cashout_txns']
            names = ['Mean Amount', 'Mean Balance', 'Transactions', 'Cashout Transactions']
            feature_keys, names = check_feature_existence(feature_keys, names, self.features['mobilemoney'])
            distributions_plot(self.features['mobilemoney'], feature_keys, names, color='orchid')
            plt.savefig(self.outputs_path / 'plots' / 'mobilemoney.png', dpi=300)
            plt.show()

        # Spatial plots
        if self.features['location'] is not None:
            for shapefile_name in self.ds.shapefiles.keys():
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                columns = [c for c in self.features['location'].columns if
                           shapefile_name in c and 'percent' not in c and 'Unknown' not in c]
                
                # Sum columns and convert to pandas
                location_pd = self.features['location'].compute()
                counts = location_pd[columns].sum().to_frame()
                counts.columns = ['txn_count']
                counts.index = ['_'.join(c.split('_')[2:])[:-1] for c in counts.index]
                counts['region'] = counts.index
                counts = self.ds.shapefiles[shapefile_name].merge(counts, on='region', how='left')
                counts['txn_count'] = counts['txn_count'].fillna(0) / counts['txn_count'].sum()
                counts.plot(ax=ax, column='txn_count', cmap='magma', legend=True, legend_kwds={'shrink': 0.5})
                ax.axis('off')
                ax.set_title('Proportion of Transactions by ' + shapefile_name, fontsize='large')
                plt.tight_layout()
                plt.savefig(self.outputs_path / 'plots' / f'spatial_{shapefile_name}.png')
                plt.show()

        # Cuts by feature usage (mobile money, mobile data, international calls)
        if self.features['cdr'] is not None:

            all_subscribers = self.features['cdr'][['name']]

            if self.features['international'] is not None:
                international_subscribers: Optional[DaskDataFrame] = self.features['international'][
                    self.features['international']['international_all__recipient_id__count'] > 0
                ][['name']]
            else:
                international_subscribers = None

            if self.features['mobiledata'] is not None:
                mobiledata_subscribers: Optional[DaskDataFrame] = self.features['mobiledata'][
                    self.features['mobiledata']['mobiledata_num_transactions'] > 0
                ][['name']]
            else:
                mobiledata_subscribers = None

            if self.features['mobilemoney'] is not None:
                mobilemoney_subscribers: Optional[DaskDataFrame] = self.features['mobilemoney'][
                    self.features['mobilemoney']['mobilemoney_all_all_txns'] > 0
                ][['name']]
            else:
                mobilemoney_subscribers = None

            features = [active_days_key, mean_call_duration_key, number_of_antennas_key]
            names = ['Active Days', 'Mean Call Duration', 'Number of Antennas']

            fig, ax = plt.subplots(1, len(features), figsize=(20, 5))
            for a in range(len(features)):
                boxplot = []
                for subscribers, slice_name in [(all_subscribers, 'All'),
                                                (international_subscribers, 'I Callers'),
                                                (mobiledata_subscribers, 'MD Users'),
                                                (mobilemoney_subscribers, 'MM Users')]:
                    if subscribers is not None:
                        users = self.features['cdr'].merge(subscribers, how='inner', on='name')
                        slice = users[['name', features[a]]].compute()
                        slice['slice_name'] = slice_name
                        boxplot.append(slice)
                boxplot_df = pd.concat(boxplot)
                boxplot_df[features[a]] = boxplot_df[features[a]].astype('float')
                sns.boxplot(data=boxplot_df, x=features[a], y='slice_name', ax=ax[a], palette="Set2", orient='h')
                ax[a].set_xlabel('Feature')
                ax[a].set_ylabel(names[a])
                ax[a].set_title(names[a], fontsize='large')
                clean_plot(ax[a])
            plt.savefig(self.outputs_path / 'plots' / 'boxplots.png', dpi=300)
            plt.show()
