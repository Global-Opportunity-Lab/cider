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

import glob
import os
import shutil
from collections import defaultdict
from typing import Dict, Optional, Union

import geopandas as gpd  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd
import rasterio  # type: ignore[import]
from helpers.plot_utils import voronoi_tessellation
from helpers.satellite_utils import quadkey_to_polygon
from helpers.utils import get_spark_session, make_dir
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from rasterio.mask import mask  # type: ignore[import]
from rasterio.merge import merge  # type: ignore[import]
from shapely.geometry import mapping  # type: ignore[import]

from .datastore import DataStore, DataType


class Satellite:

    def __init__(self,
                 datastore: DataStore,
                 dataframes: Optional[Dict[str, Optional[Union[PandasDataFrame, SparkDataFrame]]]] = None,
                 clean_folders: bool = False):
        self.cfg = datastore.cfg
        self.ds = datastore
        self.outputs = datastore.working_directory_path / 'satellite'
        # Prepare working directories
        make_dir(self.outputs, clean_folders)
        make_dir(self.outputs / 'outputs')
        make_dir(self.outputs / 'maps')
        make_dir(self.outputs / 'tables')

        # Spark setup
        spark = get_spark_session(self.cfg)
        self.spark = spark

        # Load data into datastore
        dataframes = dataframes if dataframes else defaultdict(lambda: None)
        data_type_map = {DataType.ANTENNAS: dataframes['antennas'],
                         DataType.SHAPEFILES: None,
                         DataType.RWI: None}
        self.ds.load_data(data_type_map=data_type_map)

    def aggregate_scores(self,  geo: str, dataset: str = 'rwi') -> None:
        """
        Aggregates wealth index contained in a raster dataset, like the Relative Wealth Index, to a certain geographic
        level, taking into account population density levels.

        Args:
            geo: Geographic level of aggregation: the corresponding antennas or admin boundaries shapefiles have to have
                been loaded.
            dataset: Which wealth/income map to use - only 'rwi' for now.
        """
        # Check data is loaded and preprocess it
        if dataset == 'rwi':
            if self.ds.rwi is None:
                raise ValueError("The RWI data has not been loaded.")
            scores = self.ds.rwi
            scores = scores.rename(columns={'rwi': 'score'})
            scores['polygon'] = scores['quadkey'].map(quadkey_to_polygon)
        else:
            raise NotImplementedError(f"'{dataset}' scores are not supported yet.")

        # Obtain shapefiles for masking of raster data
        if geo in ['antenna_id', 'tower_id']:
            if self.ds.antennas is None:
                raise ValueError("Antennas have not been loaded!")
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

        # Read raster with population data, mask with scores and create new score band, write multiband output
        pop_fpath = self.cfg.input_data.file_paths.population
        pop_score_fpath = self.outputs / 'pop_{dataset}.tif'
        if not os.path.isfile(pop_score_fpath):
            temp_folder = self.outputs / 'temp'
            make_dir(temp_folder, remove=True)
            with rasterio.open(pop_fpath) as src:
                meta = src.meta
                for i, row in scores.iterrows():
                    row = row.copy()
                    score = row['score']
                    geoms = [mapping(row['polygon'])]

                    # Mask with shape, create new multiband image
                    out_image, out_transform = mask(src, geoms, crop=True)
                    out_image = np.nan_to_num(out_image)[0]
                    new_band = np.full_like(out_image, fill_value=score)
                    new_band = np.where(out_image == 0, 0, new_band)

                    # Update metadata
                    new_meta = meta.copy()
                    new_meta.update({'transform': out_transform,
                                     'count': 2,
                                     'height': out_image.shape[0],
                                     'width': out_image.shape[1],
                                     'nodata': 0})

                    # Write to out file
                    with rasterio.open(temp_folder / f'{i}.tif', 'w', **new_meta) as dst:
                        for idx, band in enumerate([out_image, new_band]):
                            dst.write_band(idx + 1, band)

            # Open one raster to get metadata
            new_raster_paths = glob.glob(temp_folder / '*.tif')
            src = rasterio.open(new_raster_paths[0])

            # Merge all rasters
            mosaic, out_trans = merge(new_raster_paths)

            # Update the metadata
            out_meta = src.meta.copy()
            out_meta.update({"height": mosaic.shape[1],
                             "width": mosaic.shape[2],
                             "transform": out_trans})

            # Write mosaic to disk
            with rasterio.open(pop_score_fpath, "w", **out_meta) as dest:
                dest.write(mosaic)

        # Remove folder with intermediate rasters
        shutil.rmtree(self.outputs / 'temp', ignore_errors=True)

        # Read raster with population and score bands, mask with admin shapes and aggregate
        out_data = []
        with rasterio.open(pop_score_fpath) as src:
            for _, row in shapes.iterrows():
                idx = row[geo]
                geometry = row['geometry']
                geoms = [mapping(row['geometry'])]
                out_image, out_transform = mask(src, geoms, crop=True)
                out_image = np.nan_to_num(out_image)
                pop, score = out_image[0], out_image[1]
                score = (score * pop).sum() / pop.sum()
                total_pop = pop.sum()
                out_data.append([idx, geometry, score, total_pop])
        df = gpd.GeoDataFrame(pd.DataFrame(data=out_data, columns=['region', 'geometry', 'score', 'pop']),
                              geometry='geometry')

        # Save shapefile
        df.to_file(self.outputs / 'maps' / f'{geo}_{dataset}.geojson', driver='GeoJSON')

        # Plot map
        fig, ax = plt.subplots(1, figsize=(10, 10))

        df.plot(ax=ax, color='lightgrey')
        df.plot(ax=ax, column='score', cmap='RdYlGn', legend=True, legend_kwds={'shrink': 0.5})

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outputs / 'maps' / f'{geo}_{dataset}.png', dpi=300)
        plt.show()
