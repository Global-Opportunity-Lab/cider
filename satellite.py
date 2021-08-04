from box import Box
import geopandas as gpd
from helpers.io_utils import *
from helpers.plot_utils import *
from helpers.satellite_utils import *
from helpers.utils import *
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import yaml


class SatellitePredictor:

    def __init__(self, cfg_dir: str, dataframes: dict = None, clean_folders: bool = False):

        # Read config file
        with open(cfg_dir, "r") as ymlfile:
            cfg = Box(yaml.load(ymlfile),  Loader=yaml.FullLoader)
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

    def aggregate_scores(self, dataset: str = 'rwi'):
        # Check data is loaded and preprocess it
        if dataset == 'rwi':
            if not self.rwi:
                raise ValueError("The RWI data has not been loaded.")
            scores = self.rwi
            scores = scores.rename(columns={'rwi': 'score'})
            scores['polygon'] = scores.apply(quadkey_to_polygon, axis=1)
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

        # Read raster with population data, mask with scores, compute number of people living in each quadkey polygon
        raster_fpath = self.data + self.file_names.population
        out_data = []
        with rasterio.open(raster_fpath) as src:
            for _, row in scores.iterrows():
                row = row.copy()
                geoms = [mapping(row['polygon'])]
                out_image, out_transform = mask(src, geoms, crop=True)
                out_image = np.nan_to_num(out_image)
                row['pop'] = out_image.sum()
                out_data.append(row)
        scores_pop = gpd.GeoDataFrame(pd.DataFrame(data=out_data, columns=list(scores.columns) + ['pop']))

        # Compute population-weighted scores per admin unit
        shapes = gpd.GeoDataFrame(shapes)
        scores_pop['unit_area'] = scores['polygon'].area
        intersection = gpd.overlay(scores_pop, shapes, how='intersection')
        intersection['area'] = intersection['geometry'].area
        intersection['pop_rel'] = intersection['pop'] * intersection['area'] / intersection['unit_area']
        wm = lambda x: np.average(x, weights=intersection.loc[x.index, "pop_rel"])
        intersection = intersection[intersection['pop_rel'] != 0]
        df = intersection.groupby(self.geo).agg({'score': wm, 'pop_rel': 'sum'}).reset_index()

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
