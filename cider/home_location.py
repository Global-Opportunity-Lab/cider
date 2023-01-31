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

# TODO: Implement weights for ground truth
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union

import geopandas as gpd  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd
import rasterio  # type: ignore[import]
from helpers.plot_utils import voronoi_tessellation
from helpers.utils import get_spark_session, make_dir
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import (col, count, countDistinct, desc_nulls_last,
                                   hour, row_number)
from pyspark.sql.window import Window
from rasterio.mask import mask  # type: ignore[import]
from shapely.geometry import mapping  # type: ignore[import]

from .datastore import DataStore, DataType


class HomeLocator:

    def __init__(self,
                 datastore: DataStore,
                 dataframes: Optional[Dict[str, Optional[Union[PandasDataFrame, SparkDataFrame]]]] = None,
                 clean_folders: bool = False):
        self.cfg = datastore.cfg
        self.ds = datastore
        self.outputs = datastore.outputs + 'homelocation/'

        # Initialize values
        self.user_id = 'subscriber_id'
        self.home_locations: Dict[Tuple[str, str], PandasDataFrame] = {}
        self.accuracy_tables: Dict[Tuple[str, str], PandasDataFrame] = {}

        # Prepare working directories
        make_dir(self.outputs, clean_folders)
        make_dir(self.outputs + '/outputs/')
        make_dir(self.outputs + '/maps/')
        make_dir(self.outputs + '/tables/')

        # Spark setup
        spark = get_spark_session(self.cfg)
        self.spark = spark

        # Load data into datastore
        dataframes = dataframes if dataframes else defaultdict(lambda: None)
        data_type_map = {DataType.CDR: dataframes['cdr'],
                         DataType.ANTENNAS: dataframes['antennas'],
                         DataType.SHAPEFILES: None,
                         DataType.HOME_GROUND_TRUTH: None,
                         DataType.POVERTY_SCORES: None}
        self.ds.load_data(data_type_map=data_type_map)

        # Clean and merge CDR data
        outgoing = (self.ds.cdr
                    .select(['caller_id', 'caller_antenna', 'timestamp', 'day'])
                    .withColumnRenamed('caller_id', 'subscriber_id')
                    .withColumnRenamed('caller_antenna', 'antenna_id'))
        incoming = (self.ds.cdr
                    .select(['recipient_id', 'recipient_antenna', 'timestamp', 'day'])
                    .withColumnRenamed('recipient_id', 'subscriber_id')
                    .withColumnRenamed('recipient_antenna', 'antenna_id'))
        self.ds.cdr = (outgoing
                       .select(incoming.columns)
                       .union(incoming)
                       .na.drop()
                       .withColumn('hour', hour('timestamp')))

        # Filter CDR to only desired hours
        if self.ds.filter_hours is not None:
            self.ds.cdr = self.ds.cdr.where(col('hour').isin(self.ds.filter_hours))

    def get_home_locations(self, geo: str, algo: str = 'count_transactions') -> PandasDataFrame:
        """
        Infer home locations of users based on their CDR transactions

        Args:
            geo: geographic level at which to compute home locations; must be antenna_id, tower_id (if available), or
                one of the levels provided by the loaded shapefiles
            algo: algorithm to use, to be chosen from ['count_transactions', 'count_days', count_modal_days]

        Returns: pandas df with inferred home location for each user
        """
        # Get tower ID for each transaction
        if geo == 'tower_id':
            if 'tower_id' not in self.ds.cdr.columns:
                self.ds.cdr = (self.ds.cdr
                               .join(self.ds.antennas
                                     .select(['antenna_id', 'tower_id']).na.drop(), on='antenna_id', how='inner'))

        # Get polygon for each transaction based on antenna latitude and longitudes
        elif geo in self.ds.shapefiles.keys():
            if geo not in self.ds.cdr.columns:
                antennas_df = self.ds.antennas.na.drop().toPandas()
                antennas = gpd.GeoDataFrame(antennas_df,
                                            geometry=gpd.points_from_xy(antennas_df['longitude'],
                                                                        antennas_df['latitude']))
                antennas.crs = {"init": "epsg:4326"}
                antennas = gpd.sjoin(antennas, self.ds.shapefiles[geo], op='within', how='left')[
                    ['antenna_id', 'region']].rename({'region': geo}, axis=1)
                antennas = self.spark.createDataFrame(antennas.dropna())
                length_before = self.ds.cdr.count()
                self.ds.cdr = self.ds.cdr.join(antennas, on='antenna_id', how='inner')
                length_after = self.ds.cdr.count()
                if length_before != length_after:
                    print('Warning: %i (%.2f percent of) transactions not located in a polygon' %
                          (length_before - length_after, 100 * (length_before - length_after) / length_before))

        elif geo != 'antenna_id':
            raise ValueError('Invalid geography, must be antenna_id, tower_id, or shapefile name')

        if algo == 'count_transactions':
            grouped = self.ds.cdr.groupby([self.user_id, geo]).agg(count('timestamp').alias('count_transactions'))
            window = Window.partitionBy(self.user_id).orderBy(desc_nulls_last('count_transactions'))
            grouped = grouped.withColumn('order', row_number().over(window))\
                .where(col('order') == 1)\
                .select([self.user_id, geo, 'count_transactions'])
        
        elif algo == 'count_days':
            grouped = self.ds.cdr.groupby([self.user_id, geo]).agg(countDistinct('day').alias('count_days'))
            window = Window.partitionBy(self.user_id).orderBy(desc_nulls_last('count_days'))
            grouped = grouped.withColumn('order', row_number().over(window))\
                .where(col('order') == 1)\
                .select([self.user_id, geo, 'count_days'])

        elif algo == 'count_modal_days':
            grouped = self.ds.cdr.groupby([self.user_id, 'day', geo])\
                .agg(count('timestamp').alias('count_transactions_per_day'))
            window = Window.partitionBy([self.user_id, 'day']).orderBy(desc_nulls_last('count_transactions_per_day'))
            grouped = grouped.withColumn('order', row_number().over(window))\
                .where(col('order') == 1)\
                .groupby([self.user_id, geo])\
                .agg(count('order').alias('count_modal_days'))
            window = Window.partitionBy([self.user_id]).orderBy(desc_nulls_last('count_modal_days'))
            grouped = grouped.withColumn('order', row_number().over(window))\
                .where(col('order') == 1)\
                .select([self.user_id, geo, 'count_modal_days'])

        else:
            raise ValueError('Home location algorithm not recognized. Must be one of count_transactions, count_days, '
                             'or count_modal_days')

        grouped_df = grouped.toPandas()
        grouped_df.to_csv(self.outputs + '/outputs/' + geo + '_' + algo + '.csv', index=False)
        self.home_locations[(geo, algo)] = grouped_df
        return grouped_df

    def accuracy(self, geo: str, algo: str = 'count_transactions') -> PandasDataFrame:
        """
        Use ground truth data on users' homes to compute accuracy metrics for the inferred home locations

        Args:
            geo: compute accuracy metrics at geographic level specified by argument 'geo'
            algo: use home locations inferred using algo

        Returns: accuracy table (accuracy, precision, recall) as pandas df
        """

        if self.ds.home_ground_truth is None:
            raise ValueError('Ground truth dataset must be loaded to calculate accuracy statistics.')

        if (geo, algo) not in self.home_locations:
            raise ValueError(f"Home locations at {geo} geo level using algo {algo} have not been computed yet!")
        
        # Inner join ground truth data and inferred home locations
        merged = (self.home_locations[(geo, algo)]
                  .rename({geo: geo + '_inferred'}, axis=1)
                  .merge(self.ds.home_ground_truth.rename({geo: geo + '_groundtruth'}, axis=1),
                         on=self.user_id, how='inner'))
        print('Observations with inferred home location: %i (%i unique)' %
              (len(self.home_locations[(geo, algo)]), len(self.home_locations[(geo, algo)][self.user_id].unique())))
        print('Observations with ground truth home location: %i (%i unique)' %
              (len(self.ds.home_ground_truth), len(self.ds.home_ground_truth[self.user_id].unique())))
        print('Observations with both: %i (%i unique)' % (len(merged), len(merged[self.user_id].unique())))

        # Correct observations are ones where the groundtruth home location is the same as the inferred home location
        merged['correct'] = merged[geo + '_inferred'] == merged[geo + '_groundtruth']

        # Calculate overall accuracy (percent of observations correctly located)
        overall_accuracy = merged['correct'].mean()
        print('Overall accuracy: %.2f' % overall_accuracy)

        # Calculate precision and recall for each antenna/tower/polygon
        recall = merged.rename({geo + '_groundtruth': geo, 'correct': 'recall'}, axis=1)[[geo, 'recall']]\
            .groupby(geo, as_index=False).agg('mean')
        precision = merged.rename({geo + '_inferred': geo, 'correct': 'precision'}, axis=1)[[geo, 'precision']]\
            .groupby(geo, as_index=False).agg('mean')
        table = recall.merge(precision, on=geo, how='outer').fillna(0).sort_values('precision', ascending=False)
        table['overall_accuracy'] = overall_accuracy

        # Save table
        table.to_csv(self.outputs + '/tables/' + geo + '_' + algo + '.csv', index=False)
        self.accuracy_tables[(geo, algo)] = table
        return table

    def map(self, geo: str, algo: str = 'count_transactions', kind: str = 'population', voronoi: bool = False) -> None:
        """
        Plot distribution of homes, accuracy of home locations, or average poverty score (predicted using the ML model
        or otherwise) on a map

        Args:
            geo: plot at geographic level specified by argument 'geo'
            algo: algorithm responsible for the home inference
            kind: indicator to plot - ['precision', 'recall', 'poverty']
            voronoi: whether to use voronoi tessellation when plotting at the antenna/tower level
        """

        if self.ds.antennas is None:
            raise ValueError('Antennas must be loaded to construct maps.')

        if (geo, algo) not in self.home_locations:
            raise ValueError(f"Home locations at {geo} geo level using algo {algo} have not been computed yet!")
        
        if kind not in ['population', 'poverty', 'precision', 'recall']:
            raise ValueError('Map types are population, poverty, precision, and recall')
            
        if kind == 'poverty' and self.ds.poverty_scores is None:
            raise ValueError('Poverty scores must be loaded to construct poverty map.')

        if kind in ['precision', 'recall'] and self.accuracy_tables == {}:
            raise ValueError('Accuracy must be calculated to construct accuracy map.')

        # Get population assigned to each antenna/tower/polygon
        population = self.home_locations[(geo, algo)].groupby(geo).agg('count')\
            .rename({self.user_id: 'population'}, axis=1)
        
        # For poverty map, get average polygon for subscribers assigned to each antenna/tower/polygon and merge with
        # population data
        if kind == 'poverty':
            poverty = (self.home_locations[(geo, algo)]
                       .merge(self.ds.poverty_scores.rename({'name': self.user_id}, axis=1),
                              on=self.user_id, how='inner'))
            poverty = poverty.groupby(geo).agg('mean').rename({'predicted': 'poverty'}, axis=1)
            population = population.merge(poverty, on=geo, how='left')
        
        # If accuracy map, merge accuracy data with population data
        elif kind in ['precision', 'recall']:
            population = population.merge(self.accuracy_tables[(geo, algo)], on=geo, how='left')

        if geo in ['antenna_id', 'tower_id']:

            # Get pandas dataframes of antennas/towers
            if geo == 'antenna_id':
                points = self.ds.antennas.toPandas().dropna(subset=['antenna_id', 'latitude', 'longitude'])
            else:
                points = self.ds.antennas.toPandas()[['tower_id', 'latitude', 'longitude']]\
                    .dropna().drop_duplicates().copy()

            # Calculate voronoi tesselation and merge to population data
            if voronoi:
                if len(self.ds.shapefiles.keys()) == 0:
                    raise ValueError('At least one shapefile must be loaded to compute voronoi polygons.')
                voronoi_polygons = voronoi_tessellation(points, list(self.ds.shapefiles.values())[0], key=geo)
                population = voronoi_polygons.merge(population, on=geo, how='left')
            
            # If not voronoi, create geodataframe of latitude/longitude coordinates and merge to population data
            else:
                points = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points['longitude'], points['latitude']))
                population = points.merge(population, on=geo, how='left')
        
        # If polygons, merge polygon shapefile to population data
        elif geo in self.ds.shapefiles.keys():
            population = self.ds.shapefiles[geo].rename({'region': geo}, axis=1).merge(population, on=geo, how='left')

        else:
            raise ValueError('Invalid geometry.')

        # Null population after left join means 0 population assigned to antenna/tower/polygon
        population['population'] = population['population'].fillna(0)

        # Save shapefile
        population.to_file(self.outputs + '/maps/' + geo + '_' + algo + '_' + kind + '_voronoi' +
                           str(voronoi) + '.geojson', driver='GeoJSON')

        # Normalize population data for map
        population['population'] = population['population']/population['population'].sum()
        
        # Create map
        fig, ax = plt.subplots(1, figsize=(10, 10))

        if geo in ['antenna_id', 'tower_id'] and voronoi is False:
            
            # If points map and shapefile loaded, plot shapefile as background to map
            if len(self.ds.shapefiles.keys()) > 0:
                list(self.ds.shapefiles.values())[0].plot(ax=ax, color='lightgrey')
            
            # Plot points, sized by population and colored by outcome of interest. Plot points with no
            # population/outcome in light grey.
            population.plot(ax=ax, column=kind, markersize=population['population']*10000,
                            legend=True, legend_kwds={'shrink': 0.5})
            population[(pd.isnull(population[kind])) | (population['population'] == 0)]\
                .plot(ax=ax, color='grey', markersize=10, legend=False)

        else:

            # Plot polygons, colored by outcome of interest. Plot polygons with no outcome in light grey.
            population.plot(ax=ax, color='lightgrey')
            population.plot(ax=ax, column=kind, legend=True, legend_kwds={'shrink': 0.5})

        # Clean and save plot
        ax.axis('off')
        title = 'Population Map' if kind == 'population' else 'Poverty Map' if kind == 'poverty' else 'Precision Map' \
            if kind == 'precision' else 'Recall Map'
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(self.outputs + '/maps/' + geo + '_' + algo + '_' + kind + '_voronoi' + str(voronoi) + '.png',
                    dpi=300)
        plt.show()

    def pop_comparison(self, geo: str, algo: str = 'count_transactions') -> None:
        """
        Use a population density raster - e.g. FB D4G's high resolution density maps - to compare the distribution of
        inferred home locations to the population

        Args:
            geo: compare pop distribution at geographic level specified by argument 'geo'
            algo: algorithm responsible for the home inference
        """
        # Get population assigned to each antenna/tower/polygon
        if (geo, algo) in self.home_locations:
            homes = self.home_locations[(geo, algo)].groupby(geo).agg('count')\
                                                  .rename({self.user_id: 'population'}, axis=1).reset_index()
        else:
            raise ValueError(f"Home locations at {geo} geo level using algo {algo} have not been computed yet!")

        # Obtain shapefiles for masking of raster data
        if geo in ['antenna_id', 'tower_id']:
            # Get pandas dataframes of antennas/towers
            if geo == 'antenna_id':
                points = self.ds.antennas.toPandas().dropna(subset=['antenna_id', 'latitude', 'longitude'])
            else:
                points = self.ds.antennas.toPandas()[
                    ['tower_id', 'latitude', 'longitude']].dropna().drop_duplicates().copy()

            # Calculate voronoi tesselation
            if len(self.ds.shapefiles.keys()) == 0:
                raise ValueError('At least one shapefile must be loaded to compute voronoi polygons.')
            shapes = voronoi_tessellation(points, list(self.ds.shapefiles.values())[0], key=geo)

        elif geo in self.ds.shapefiles.keys():
            shapes = self.ds.shapefiles[geo].rename({'region': geo}, axis=1)
        else:
            raise ValueError('Invalid geometry.')

        # Read raster with population data, mask with each shape and compute population in units
        raster_fpath = self.cfg.path.input_data.file_paths.population
        out_data = []
        with rasterio.open(raster_fpath) as src:
            for _, row in shapes.iterrows():
                idx = row[geo]
                geometry = row['geometry']
                geoms = [mapping(row['geometry'])]
                out_image, out_transform = mask(src, geoms, crop=True)
                out_image = np.nan_to_num(out_image)
                out_data.append([idx, geometry, out_image.sum()])
        population = pd.DataFrame(data=out_data, columns=['region', 'geometry', 'pop'])

        # Merge with number of locations and compute pct point difference of relative populations
        homes = homes.rename(columns={geo: 'region', 'population': 'homes'})
        df = pd.merge(population, homes[['region', 'homes']], on='region', how='left')
        df = df.fillna(0)

        df['pop_pct'] = df['pop'] / df['pop'].sum() * 100
        df['homes_pct'] = df['homes'] / df['homes'].sum() * 100
        df['diff'] = df['homes_pct'] - df['pop_pct']
        df = gpd.GeoDataFrame(df)

        # Save shapefile
        df.to_file(self.outputs + '/maps/' + geo + '_' + algo + '_' + 'pop_comparisons' + '.geojson', driver='GeoJSON')

        # Plot map
        fig, ax = plt.subplots(1, figsize=(10, 10))

        df.plot(ax=ax, color='lightgrey')
        df.plot(ax=ax, column='diff', cmap='RdYlGn', legend=True, legend_kwds={'shrink': 0.5})

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outputs + '/maps/' + geo + '_' + algo + '_' + 'pop_comparisons' + '.png', dpi=300)
        plt.show()
