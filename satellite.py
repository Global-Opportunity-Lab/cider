from box import Box  # type: ignore[import]
import geopandas as gpd  # type: ignore[import]
import glob
from helpers.io_utils import load_antennas, load_shapefile
from helpers.plot_utils import voronoi_tessellation
from helpers.satellite_utils import quadkey_to_polygon
from helpers.utils import get_spark_session, make_dir
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import os
import pandas as pd  # type: ignore[import]
from pyspark.sql import DataFrame as SparkDataFrame  # type: ignore[import]
import rasterio  # type: ignore[import]
from rasterio.mask import mask  # type: ignore[import]
from rasterio.merge import merge  # type: ignore[import]
from shapely.geometry import mapping  # type: ignore[import]
from typing import Dict, Optional
import yaml


class SatellitePredictor:

    def __init__(self,
                 cfg_dir: str,
                 dataframes: Optional[Dict[str, SparkDataFrame]] = None,
                 clean_folders: bool = False) -> None:

        # Read config file
        with open(cfg_dir, "r") as ymlfile:
            cfg = Box(yaml.load(ymlfile,  Loader=yaml.FullLoader))
        self.cfg = cfg
        data = cfg.path.satellite.data
        self.data = data
        outputs = cfg.path.satellite.outputs
        self.outputs = outputs
        file_names = cfg.path.satellite.file_names
        self.file_names = file_names

        # Initialize values
        self.geo = cfg.col_names.geo

        # Prepare working directory
        make_dir(outputs, clean_folders)
        make_dir(outputs + '/outputs/')
        make_dir(outputs + '/maps/')
        make_dir(outputs + '/tables/')
        
        # Spark setup
        spark = get_spark_session(cfg)
        self.spark = spark

        # Load antennas data 
        dataframe = dataframes['antennas'] if dataframes is not None and 'antennas' in dataframes.keys() else None
        fpath = data + file_names.antennas if file_names.antennas is not None else None
        if file_names.antennas is not None or dataframe is not None:
            print('Loading antennas...')
            self.antennas = load_antennas(self.cfg, fpath, df=dataframe)
        else:
            self.antennas = None

        # Load poverty predictions
        if self.file_names.rwi:
            self.rwi = pd.read_csv(self.data + self.file_names.rwi, dtype={'quadkey': str})
        else:
            self.rwi = None

        # Load shapefiles
        self.shapefiles = {}
        shapefiles = file_names.shapefiles
        for shapefile_fname in shapefiles.keys():
            self.shapefiles[shapefile_fname] = load_shapefile(data + shapefiles[shapefile_fname])

    def aggregate_scores(self, dataset: str = 'rwi') -> None:
        """
        Aggregate poverty scores contained in a raster dataset, like the Relative Wealth Index, to a certain geographic
        level, taking into account population density levels

        Args:
            dataset: which poverty scores to use - only 'rwi' for now
        """
        # Check data is loaded and preprocess it
        if dataset == 'rwi':
            if self.rwi is None:
                raise ValueError("The RWI data has not been loaded.")
            scores = self.rwi
            scores = scores.rename(columns={'rwi': 'score'})
            scores['polygon'] = scores['quadkey'].map(quadkey_to_polygon)
        else:
            raise NotImplementedError("'{scores}' scores are not supported yet.")

        # Obtain shapefiles for masking of raster data
        if self.geo in ['antenna_id', 'tower_id']:
            # Get pandas dataframes of antennas/towers
            if self.geo == 'antenna_id':
                points = self.antennas.toPandas().dropna(subset=['antenna_id', 'latitude', 'longitude'])
            else:
                points = self.antennas.toPandas()[
                    ['tower_id', 'latitude', 'longitude']].dropna().drop_duplicates().copy()

            # Calculate voronoi tesselation
            if len(self.shapefiles.keys()) == 0:
                raise ValueError('At least one shapefile must be loaded to compute voronoi polygons.')
            shapes = voronoi_tessellation(points, list(self.shapefiles.values())[0], key=self.geo)

        elif self.geo in self.shapefiles.keys():
            shapes = self.shapefiles[self.geo].rename({'region': self.geo}, axis=1)
        else:
            raise ValueError('Invalid geometry.')

        # Read raster with population data, mask with scores and create new score band, write multiband output
        pop_fpath = self.data + self.file_names.population
        pop_score_fpath = self.outputs + f'/pop_{dataset}.tif'
        if not os.path.isfile(pop_score_fpath):
            temp_folder = self.outputs + '/temp'
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
                    with rasterio.open(temp_folder + f'/{i}.tif', 'w', **new_meta) as dst:
                        for idx, band in enumerate([out_image, new_band]):
                            dst.write_band(idx + 1, band)

            new_raster_paths = glob.glob(temp_folder + '/*.tif')
            for raster in new_raster_paths:
                src = rasterio.open(raster)
                break

            mosaic, out_trans = merge(new_raster_paths)

            # Update the metadata
            out_meta = src.meta.copy()
            out_meta.update({"height": mosaic.shape[1],
                             "width": mosaic.shape[2],
                             "transform": out_trans})

            with rasterio.open(pop_score_fpath, "w", **out_meta) as dest:
                dest.write(mosaic)

        # Read raster with population and score bands, mask with admin shapes and aggregate
        out_data = []
        with rasterio.open(pop_score_fpath) as src:
            for _, row in shapes.iterrows():
                idx = row[self.geo]
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
        df.to_file(self.outputs + '/maps/' + self.geo + '_' + dataset + '.geojson', driver='GeoJSON')

        # Plot map
        fig, ax = plt.subplots(1, figsize=(10, 10))

        df.plot(ax=ax, color='lightgrey')
        df.plot(ax=ax, column='score', cmap='RdYlGn', legend=True, legend_kwds={'shrink': 0.5})

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outputs + '/maps/' + self.geo + '_' + dataset + '.png', dpi=300)
        plt.show()
