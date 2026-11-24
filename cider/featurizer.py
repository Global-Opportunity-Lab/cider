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
from typing import Any, Dict, Optional, Union

import dask
import dask.dataframe as dd
import geopandas as gpd  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns  # type: ignore[import]

from helpers.features import all_dask, prepare_cdr_base
from helpers.legacy_feature_names import rename_cdr_features_to_legacy
from helpers.plot_utils import clean_plot, dates_xaxis, distributions_plot
from helpers.utils import (cdr_bandicoot_format, filter_by_phone_numbers_to_featurize,
                           get_dask_client, long_join_pandas, long_join_dask, make_dir,
                           read_parquet, save_parquet, save_df)
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
        make_dir(outputs_path / 'datasets')

        self.features: Dict[str, Optional[DaskDataFrame]] = {'cdr': None, 'international': None, 'recharges': None,
                                                              'location': None, 'mobiledata': None, 'mobilemoney': None}

        # Dask setup
        client = get_dask_client(self.cfg)
        self.client = client
        dask_cfg = self.cfg.get('dask', {})
        n_workers = dask_cfg.get('n_workers', 4)
        n_threads = dask_cfg.get('threads_per_worker', 1)
        dask_cfg['num_partitions'] = dask_cfg.get('num_partitions', 16)
        self.dask_cfg = dask_cfg
        print(f'Dask: {n_workers} worker(s), {n_threads} thread(s) per worker, {dask_cfg["num_partitions"]} partitions (total parallelism: {n_workers * n_threads})')
        if hasattr(client, 'dashboard_link') and client.dashboard_link:
            print('Dask dashboard:', client.dashboard_link)
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
            self.output_format = _OutputFormat.PARQUET


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
        Compute CDR features and save to disk. Delegates to the Dask-based implementation.
        The bc_chunksize and bc_processes arguments are ignored (kept for API compatibility).
        """
        self.cdr_features_dask()

    def cdr_features_dask(self) -> None:
        """
        Compute CDR features using dask and save to disk
        """
        # Check that CDR is present to calculate international features
        if self.ds.cdr is None:
            raise ValueError('CDR file must be loaded to calculate CDR features.')
        print('Calculating CDR features...')

        # Restart workers to flush memory from previous feature computations.
        # Earlier features (location, mobiledata) leave data resident in worker
        # memory, pushing RAM near Dask's 80% pause threshold. CDR features
        # involve large shuffles and gathers that deadlock when workers are paused.
        try:
            from dask.distributed import get_client
            _client = get_client()
            print('Restarting Dask workers to free memory before CDR features...')
            _client.restart()
        except Exception:
            pass

        # Compute the shared CDR base (in/out duplication, filter, tag_conversations,
        # weekday/daytime) once using the distributed scheduler, then write to a temp
        # parquet directory (64 partitions). All 20 feature functions in all_dask()
        # read from those 64 small files instead of re-reading 1407 raw CDR files
        # each time. Without this cache, CDR features take 20+ hours because every
        # feature's pivot_df/.compute() traverses the full CDR task graph.
        import tempfile, shutil
        from pathlib import Path as _Path
        _temp_dir = _Path(tempfile.mkdtemp(prefix='cider_cdr_base_'))
        _cdr_base_parquet = _temp_dir / 'base'
        try:
            print('Computing CDR base (one-time read of raw files) ...')
            _cdr_base_df = prepare_cdr_base(
                self.ds.cdr,
                cfg=self.cfg.params.cdr,
                phone_numbers_to_featurize=self.phone_numbers_to_featurize,
            )
            # Run the CDR base write in the main thread. Distributed mode for this
            # 83k-task graph (read 1407 files + shuffle) creates 30+ minutes of
            # inter-worker coordination and GC pressure that eventually kills a worker.
            # Sync mode uses the full 83GB main-process RAM, runs all tasks
            # sequentially (no network overhead), and completes in a few minutes.
            # Feature functions still run distributed from the small 64-partition cache.
            with dask.config.set(scheduler='synchronous'):
                _cdr_base_df.to_parquet(
                    str(_cdr_base_parquet), write_index=False, overwrite=True, schema=None
                )
            print('CDR base written to temp parquet; computing features from cache...')
            _cdr_cached = dd.read_parquet(str(_cdr_base_parquet))

            # Use synchronous scheduler for all feature computation. Feature functions
            # call groupby().size().compute() and similar, which create repartitiontofewer
            # tasks in the distributed scheduler that kill workers via OOM. With the
            # 64-partition temp cache the sync compute is fast (seconds per feature).
            with dask.config.set(scheduler='synchronous'):
                cdr_features = all_dask(
                    _cdr_cached,
                    self.ds.antennas,
                    cfg=self.cfg.params.cdr,
                    phone_numbers_to_featurize=self.phone_numbers_to_featurize,
                    skip_base_prep=True,
                )
                # Keep all subsequent lazy operations inside the try block so the temp
                # parquet exists until save_parquet() triggers the actual compute.
                cdr_features_df = long_join_dask(cdr_features, on='caller_id', how='outer')
                cdr_features_df = cdr_features_df.rename(columns={'caller_id': 'name'})
                cdr_features_df = rename_cdr_features_to_legacy(cdr_features_df)

                if self.output_format == _OutputFormat.CSV:
                    save_df(cdr_features_df, self.outputs_path / 'datasets' / 'cdr_features_dask' / 'all', single_file=False)
                else:
                    save_parquet(cdr_features_df, self.outputs_path / 'datasets' / 'cdr_features_dask' / 'all')
                self.features['cdr'] = cdr_features_df
        finally:
            shutil.rmtree(_temp_dir, ignore_errors=True)

    # Backward compatibility
    cdr_features_spark = cdr_features_dask

    def international_features(self) -> None:
        # Check that CDR is present to calculate international features
        if self.ds.cdr is None:
            raise ValueError('CDR file must be loaded to calculate international features.')
        print('Calculating international features...')

        # Write international transactions to file — parquet (partition-parallel)
        # instead of single-file CSV so we never call df.compute() on the raw CDR
        # slice, which would trigger repartitiontofewer and OOM workers.
        international_trans = self.ds.cdr[self.ds.cdr['international'] == 'international']
        save_parquet(international_trans, self.outputs_path / 'datasets' / 'international_transactions')

        # Read back: the result is small (international txns are a fraction of CDR)
        # so a single .compute() is fine here.
        inter = read_parquet(
            self.client, self.outputs_path / 'datasets' / 'international_transactions'
        ).compute()
        inter['caller_id'] = inter['caller_id'].astype(str)

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
                # pandas 1.4 bug: as_index=False with list agg puts caller_id in the index
                # instead of columns (both empty and non-empty). reset_index() recovers it.
                expected_cols = ['caller_id'] + [name + '__' + c + '__' + ag for ag in agg]
                if len(grouped.columns) < len(expected_cols):
                    grouped = grouped.reset_index()
                grouped.columns = expected_cols
                feats.append(grouped)

        # Combine all aggregations together, write to file
        feats_df = long_join_pandas(feats, on='caller_id', how='outer').rename({'caller_id': 'name'}, axis=1)

        feats_df = filter_by_phone_numbers_to_featurize(self.phone_numbers_to_featurize, feats_df, 'name')

        feats_df.columns = [c if c == 'name' else 'international_' + c for c in feats_df.columns]

        if self.output_format == _OutputFormat.CSV:
            save_df(feats_df, self.outputs_path / 'datasets' / 'international_feats', single_file=False)
            self.features['international'] = dd.from_pandas(feats_df, npartitions=self.dask_cfg['num_partitions'])
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
            # Keep only geometry and region to avoid duplicate columns from sjoin (e.g. DISTRICT_left)
            shapefile = self.ds.shapefiles[shapefile_name][['geometry', 'region']].rename(
                columns={'region': shapefile_name}
            )
            antennas = gpd.sjoin(antennas, shapefile, predicate='within', how='left').drop('index_right', axis=1)
            antennas[shapefile_name] = antennas[shapefile_name].fillna('Unknown')
        # Keep antennas as pandas — it's a small lookup table (~thousands of rows).
        # Using a broadcast join (map_partitions) means each CDR partition merges
        # with the full antennas table locally on its worker.  This eliminates the
        # distributed shuffle join that was repartitioning CDR down to num_partitions
        # (~16) output partitions, creating one massive partition per worker and
        # triggering repartitiontofewer OOM crashes.
        antennas_pd = antennas.drop(columns=['geometry', 'latitude', 'longitude']).fillna('').reset_index(drop=True)
        antennas_pd['antenna_id'] = antennas_pd['antenna_id'].astype(str)

        cdr_bandicoot_filtered = filter_by_phone_numbers_to_featurize(
            self.phone_numbers_to_featurize, self.ds.cdr_bandicoot, 'name'
        )

        # Select only the columns needed for location features before the merge —
        # avoids carrying CDR columns (duration, direction, lat, lon, etc.) through
        # the shuffle, which multiplies per-worker memory use.
        select_cols = [c for c in ['name', 'antenna_id'] if c in cdr_bandicoot_filtered.columns]
        cdr_slim = cdr_bandicoot_filtered[select_cols].assign(
            antenna_id=cdr_bandicoot_filtered['antenna_id'].astype(str)
        )

        # Broadcast join: each CDR partition is merged with the small antennas
        # DataFrame on its own worker — no cross-worker shuffle, no repartitiontofewer.
        meta_cdr = cdr_slim._meta.merge(antennas_pd.head(0), on='antenna_id', how='left')
        cdr = cdr_slim.map_partitions(
            lambda part, ant: part.merge(ant, on='antenna_id', how='left'),
            antennas_pd,
            meta=meta_cdr,
        )

        fillna_vals = {sn: 'Unknown' for sn in self.ds.shapefiles.keys()}
        if fillna_vals:
            cdr = cdr.fillna(fillna_vals)

        # Get counts by region — write as partition-parallel parquet instead of
        # single-file CSV.  save_df(..., single_file=True) calls df.compute()
        # internally, which triggers repartitiontofewer to gather all partitions
        # onto one worker before writing — the same OOM we're avoiding above.
        for shapefile_name in self.ds.shapefiles.keys():
            if shapefile_name not in cdr.columns:
                raise ValueError(
                    f"Shapefile column '{shapefile_name}' not in CDR after merge. "
                    f"Ensure shapefile GeoDataFrame has a column renamed to '{shapefile_name}' (e.g. 'region'). "
                    f"Available columns: {list(cdr.columns)}"
                )
            gb = cdr.groupby(['name', shapefile_name]).size().compute()
            gb.name = 'count'
            out_path = self.outputs_path / 'datasets' / f'countby{shapefile_name}'
            out_path.mkdir(parents=True, exist_ok=True)
            pl.from_pandas(gb.reset_index()).write_parquet(str(out_path / '0.parquet'))

        # All aggregates are small (one row per subscriber). Compute each in Dask, then
        # join and persist in Polars — sidesteps the Dask 2025.7 optimizer reset_index bug
        # on Series (ndim=1) that fires when to_parquet triggers _simplify_up.
        unique_regions_pl = pl.from_pandas(
            cdr[['name']].drop_duplicates().compute().astype({'name': str})
        )
        for shapefile_name in self.ds.shapefiles.keys():
            rc = cdr.groupby('name')[shapefile_name].nunique().compute()
            rc.name = f'count(DISTINCT {shapefile_name})'
            unique_regions_pl = unique_regions_pl.join(
                pl.from_pandas(rc.reset_index().astype({'name': str})), on='name', how='left'
            )
        if 'tower_id' in cdr.columns:
            tc = cdr.groupby('name')['tower_id'].nunique().compute()
            tc.name = 'count(DISTINCT tower_id)'
            unique_regions_pl = unique_regions_pl.join(
                pl.from_pandas(tc.reset_index().astype({'name': str})), on='name', how='left'
            )
        out_path = self.outputs_path / 'datasets' / 'uniqueregions'
        out_path.mkdir(parents=True, exist_ok=True)
        unique_regions_pl.write_parquet(str(out_path / '0.parquet'))
        feats = unique_regions_pl.to_pandas()
        feats = feats.astype({'name': str})


        if len(self.ds.shapefiles) > 0:

            count_by_region_compiled = []
            
            # Pivot counts by region
            for shapefile_name in self.ds.shapefiles.keys():
                count_by_region = read_parquet(
                    self.client,
                    self.outputs_path / 'datasets' / f'countby{shapefile_name}'
                ).compute().astype({'name': str}).pivot(
                    index='name', columns=shapefile_name, values='count'
                ).fillna(0)

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
            self.features['location'] = dd.from_pandas(feats, npartitions=self.dask_cfg['num_partitions'])
        else:
            save_parquet(feats, self.outputs_path / 'datasets' / 'location_features')
            self.features['location'] = read_parquet(self.client, self.outputs_path / 'datasets' / 'location_features')

    def mobiledata_features(self) -> None:

        # Check that mobile internet data is loaded
        if self.ds.mobiledata is None:
            raise ValueError('Mobile data file must be loaded to calculate mobile data features.')
        print('Calculating mobile data features...')
        
        # Aggregate mobiledata use by day, to control for different definitions of "one transaction."
        # Dask shuffle uses int8 for partition indices, overflowing when npartitions > 127.  Many daily
        # files (e.g. 90 days × 2 carriers) create 180+ partitions.  repartition() here is a coalesce
        # (concatenating adjacent partitions), not a hash shuffle, so it is safe even at high partition
        # counts.  The subsequent groupby shuffle then sees ≤64 partitions (fits in int8).
        mobiledata = self.ds.mobiledata
        if mobiledata.npartitions > 64:
            mobiledata = mobiledata.repartition(npartitions=64)
        mobiledata_aggregated_by_day = mobiledata.groupby(['caller_id', 'day'])['volume'].sum().reset_index()
        
        mobiledata_aggregated_by_day = filter_by_phone_numbers_to_featurize(
            self.phone_numbers_to_featurize, mobiledata_aggregated_by_day, 'caller_id'
        )

        volume_aggs = mobiledata_aggregated_by_day.groupby('caller_id')['volume'].agg(
            ['sum', 'mean', 'min', 'max', 'std']
        ).reset_index()
        volume_aggs = volume_aggs.rename(columns={
            'sum': 'total_volume',
            'mean': 'mean_daily_volume',
            'min': 'min_daily_volume',
            'max': 'max_daily_volume',
            'std': 'std_daily_volume'
        })
        day_nunique = mobiledata_aggregated_by_day.groupby('caller_id')['day'].nunique().rename(
            'num_days'
        ).reset_index()
        feats = volume_aggs.merge(day_nunique, on='caller_id', how='left')
        feats = feats.rename(columns={'caller_id': 'name'})

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

            # Compute aggregations via Dask groupby so each worker aggregates
            # locally. Calling df.compute() first gathers all 886+ partitions
            # via repartitiontofewer, which OOMs workers under memory pressure.
            # Dask's dict-style agg does not support nunique; compute it
            # separately via SeriesGroupBy.nunique() and attach to the result.
            aggs_pd = df.groupby(['name', 'txn_type']).agg({
                'amount': ['sum', 'mean', 'min', 'max'],
                'balance_before': ['mean', 'min', 'max'],
                'balance_after': ['mean', 'min', 'max'],
                'correspondent_id': ['count']
            }).compute()
            nunique_pd = df.groupby(['name', 'txn_type'])['correspondent_id'].nunique().compute()
            aggs_pd[('correspondent_id', 'nunique')] = nunique_pd
            
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
            aggs = dd.from_pandas(aggs_pd, npartitions=self.dask_cfg['num_partitions'])
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

        # Dask groupby.agg does not support 'nunique'; compute amount aggs and day nunique separately then merge.
        # Rename to 'name' before merge so the expression graph uses 'name' consistently (avoids KeyError on
        # caller_id when to_parquet inspects divisions).
        #
        # Dask shuffle uses int8 for partition indices, overflowing when npartitions > 127.  Many daily
        # files (e.g. 90 days × 2 carriers) create 180+ partitions.  repartition() here is a coalesce
        # (concatenating adjacent partitions), not a hash shuffle, so it is safe even at high partition
        # counts.  The subsequent groupby shuffle then sees ≤64 partitions (fits in int8).
        recharges = self.ds.recharges
        if recharges.npartitions > 64:
            recharges = recharges.repartition(npartitions=64)
        amount_aggs = (
            recharges.groupby('caller_id')['amount']
            .agg(['sum', 'mean', 'min', 'max', 'count'])
            .reset_index()
            .rename(columns={'caller_id': 'name'})
        )
        day_nunique = (
            recharges.groupby('caller_id')['day']
            .nunique()
            .rename('days')
            .reset_index()
            .rename(columns={'caller_id': 'name'})
        )
        feats = amount_aggs.merge(day_nunique, on='name', how='left')
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

        features = ['cdr', 'international', 'location', 'mobiledata', 'mobilemoney', 'recharges']
        paths_to_datasets = ['cdr_features_dask/all', 'international_feats', 'location_features',
                    'mobiledata_features', 'mobilemoney_feats', 'recharges_feats']

        # Read data from disk if requested
        for feature, path_to_dataset in zip(features, paths_to_datasets):
            if self.features[feature] is None:
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
                        str(data_path / path_to_dataset / '*.csv'),
                        dtype={'caller_id': 'str', 'name': 'str'}
                    )

                elif example_file.suffix == '.parquet':
                    self.features[feature] = read_parquet(self.client, data_path / path_to_dataset)

    def _compute_household_features(self, person_features: DaskDataFrame, phone_numbers_to_featurize: DaskDataFrame) -> None:
        """
        Aggregate person-level features to household level and save to disk.

        Produces mean, std, and max for every numeric feature column, plus hh_n_persons.
        The join key from person_features is 'name' (= person_id if person_id was configured,
        otherwise phone_number).
        """
        id_col = 'person_id' if 'person_id' in phone_numbers_to_featurize.columns else 'phone_number'
        hh_mapping = phone_numbers_to_featurize[[id_col, 'household_id']].drop_duplicates()

        # Convert to Polars for efficient groupby aggregation — avoids Dask's
        # blockwisemerge OOM from joining 4 separate groupby results, and scales
        # to year-long windows without memory pressure.
        if isinstance(person_features, dd.DataFrame):
            pf_pl = pl.from_pandas(person_features.compute())
        elif isinstance(person_features, pd.DataFrame):
            pf_pl = pl.from_pandas(person_features)
        else:
            pf_pl = person_features  # already Polars

        if isinstance(hh_mapping, dd.DataFrame):
            hh_pl = pl.from_pandas(hh_mapping.compute())
        elif isinstance(hh_mapping, pd.DataFrame):
            hh_pl = pl.from_pandas(hh_mapping)
        else:
            hh_pl = hh_mapping  # already Polars

        hh_pl = hh_pl.with_columns(pl.col(id_col).cast(pl.Utf8))
        pf_pl = pf_pl.with_columns(pl.col('name').cast(pl.Utf8))

        merged = pf_pl.join(hh_pl, left_on='name', right_on=id_col, how='inner')

        feature_cols = [c for c in merged.columns if c not in ('name', 'household_id')]
        numeric_cols = [c for c in feature_cols if merged[c].dtype.is_numeric()]

        agg_exprs = []
        for c in numeric_cols:
            agg_exprs.extend([
                pl.col(c).mean().alias(f'{c}__hh_mean'),
                pl.col(c).std().alias(f'{c}__hh_std'),
                pl.col(c).max().alias(f'{c}__hh_max'),
            ])
        agg_exprs.append(pl.col('name').n_unique().alias('hh_n_persons'))

        hh_features_pl = merged.group_by('household_id').agg(agg_exprs)
        hh_features = hh_features_pl.to_pandas()

        if self.output_format == _OutputFormat.CSV:
            save_df(hh_features, self.outputs_path / 'datasets' / 'household_features', single_file=False)
        else:
            save_parquet(hh_features, self.outputs_path / 'datasets' / 'household_features')
        self.features['household'] = hh_features

    def all_features(self, read_from_disk: bool = False) -> None:
        """
        Join all feature datasets together, save to disk, and assign to attribute

        Args:
            read_from_disk: whether to load features from disk
        """
        if read_from_disk:
            # Force reload from disk to avoid mixing persisted features with in-memory
            # Dask expression graphs created earlier in this run.
            for key in ['cdr', 'international', 'location', 'mobiledata', 'mobilemoney', 'recharges']:
                self.features[key] = None
            self.load_features()

        # Recompute set of all features if it was already computed
        if 'all' in self.features.keys():
            del self.features['all']

        all_features_list = [self.features[key] for key in self.features.keys() if self.features[key] is not None]
        if all_features_list:
            # Compute each feature set to pandas individually, then merge in
            # pandas.  The previous approach repartitioned every DF to 1
            # partition before calling long_join_dask, which caused Dask to
            # use blockwisemerge — a code path that bypasses the shuffle.method
            # config and materialises the entire merged result into a single
            # task on one worker.  For large CDR windows (30-90 days of Airtel
            # data) that intermediate exceeds per-worker memory limits and also
            # fails LZ4 spill compression (>2 GB serialised object), producing
            # an infinite retry loop.
            #
            # Each individual feature set is small (one aggregated row per
            # subscriber, typically <10 MB), so computing them to pandas and
            # merging in-process is both fast and memory-safe.  We then wrap
            # the result back in a Dask DataFrame for compatibility with the
            # save/household-features code below.
            pd_frames = [
                df.compute() if isinstance(df, dd.DataFrame) else df
                for df in all_features_list
            ]
            # Cast 'name' to str on every frame — mirrors long_join_dask which
            # did df.assign(name=df['name'].astype(str)) before each merge to
            # avoid "merging string and float64 columns" errors.
            pd_frames = [df.assign(name=df['name'].astype(str)) for df in pd_frames]
            merged_pd = pd_frames[0]
            for other in pd_frames[1:]:
                merged_pd = merged_pd.merge(other, on='name', how='outer')
            # Keep as plain pandas — save_parquet and _compute_household_features
            # both handle pd.DataFrame directly.  Wrapping in dd.from_pandas causes
            # repartitiontofewer when _compute_household_features calls .compute(),
            # routing a multi-partition DF through the distributed scheduler and
            # hitting the same per-worker OOM/LZ4-spill crash we fixed above.
            all_features = merged_pd

            if self.output_format == _OutputFormat.CSV:
                save_df(all_features, self.outputs_path / 'datasets' / 'features', single_file=False)
            else:
                save_parquet(all_features, self.outputs_path / 'datasets' / 'features')
            self.features['all'] = all_features

            phone_numbers_to_featurize = getattr(self.ds, 'phone_numbers_to_featurize', None)
            if phone_numbers_to_featurize is not None and 'household_id' in phone_numbers_to_featurize.columns:
                self._compute_household_features(all_features, phone_numbers_to_featurize)
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
