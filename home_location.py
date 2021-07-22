# TODO: Implement weights for ground truth

from box import Box
import yaml
from helpers.utils import *
from helpers.plot_utils import *
from helpers.io_utils import *
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping


class HomeLocator:

    def __init__(self, cfg_dir, dataframes=None, clean_folders=False):

        # Read config file
        with open(cfg_dir, "r") as ymlfile:
            cfg = Box(yaml.load(ymlfile),  Loader=yaml.FullLoader)
        self.cfg = cfg
        data = cfg.path.home_location.data
        self.data = data
        outputs = cfg.path.home_location.outputs
        self.outputs = outputs
        file_names = cfg.path.home_location.file_names

        # Initialize values
        self.geo = cfg.col_names.geo
        self.filter_hours = cfg.params.home_location.filter_hours
        if file_names.groundtruth is not None:
            self.groundtruth = pd.read_csv(data + file_names.groundtruth)
        if file_names.poverty_scores is not None:
            self.poverty_scores = pd.read_csv(data + file_names.poverty_scores)
        self.home_locations = {}
        self.accuracy_tables = {}

        # Prepare working directory
        make_dir(outputs, clean_folders)
        make_dir(outputs + '/outputs/')
        make_dir(outputs + '/maps/')
        make_dir(outputs + '/tables/')
        
        # Spark setup
        spark = get_spark_session(cfg)
        self.spark = spark

        # Load CDR data 
        dataframe = dataframes['cdr'] if dataframes is not None and 'cdr' in dataframes.keys() else None
        fpath = data + file_names.cdr if file_names.cdr is not None else None
        if file_names.cdr is not None or dataframe is not None:
            print('Loading CDR...')
            self.cdr = load_cdr(self.cfg, fpath, df=dataframe)
        else:
            self.cdr = None

        # Load antennas data 
        dataframe = dataframes['antennas'] if dataframes is not None and 'antennas' in dataframes.keys() else None
        fpath = data + file_names.antennas if file_names.antennas is not None else None
        if file_names.antennas is not None or dataframe is not None:
            print('Loading antennas...')
            self.antennas = load_antennas(self.cfg, fpath, df=dataframe)
        else:
            self.antennas = None

        # Load shapefiles
        self.shapefiles = {}
        shapefiles = file_names.shapefiles
        for shapefile_fname in shapefiles.keys():
            self.shapefiles[shapefile_fname] = load_shapefile(data + shapefiles[shapefile_fname])

        # Clean and merge CDR data
        outgoing = (self.cdr
                    .select(['caller_id', 'caller_antenna', 'timestamp', 'day'])
                    .withColumnRenamed('caller_id', 'subscriber_id')
                    .withColumnRenamed('caller_antenna', 'antenna_id'))
        incoming = (self.cdr
                    .select(['recipient_id', 'recipient_antenna', 'timestamp', 'day'])
                    .withColumnRenamed('recipient_id', 'subscriber_id')
                    .withColumnRenamed('recipient_antenna', 'antenna_id'))
        self.cdr = (outgoing
                    .select(incoming.columns)
                    .union(incoming)
                    .na.drop()
                    .withColumn('hour', hour('timestamp')))

        # Filter CDR to only desired hours
        if self.filter_hours is not None:
            self.cdr = self.cdr.where(col('hour').isin(self.filter_hours))
        
        # Get tower ID for each transaction
        if self.geo == 'tower_id':
            self.cdr = (self.cdr
                        .join(self.antennas
                              .select(['antenna_id', 'tower_id']).na.drop(), on='antenna_id', how='inner'))
        
        # Get polygon for each transaction based on antenna latitude and longitudes
        elif self.geo in self.shapefiles.keys():
            antennas = self.antennas.na.drop().toPandas()
            antennas = gpd.GeoDataFrame(antennas, geometry=gpd.points_from_xy(antennas['longitude'], antennas['latitude']))
            antennas = gpd.sjoin(antennas, self.shapefiles[self.geo], op='within', how='left')[['antenna_id', 'region']].rename({'region':self.geo}, axis=1)
            antennas = self.spark.createDataFrame(antennas).na.drop()
            length_before = self.cdr.count()
            self.cdr = self.cdr.join(antennas, on='antenna_id', how='inner')
            length_after = self.cdr.count()
            if length_before != length_after:
                print('Warning: %i (%.2f percent of) transactions not located in a polygon' % \
                    (length_before-length_after, 100*(length_before-length_after)/length_before))

        elif self.geo != 'antenna_id':
            raise ValueError('Invalid geography, must be antenna_id, tower_id, or shapefile name')

    def filter_dates(self, start_date, end_date):

        self.cdr = filter_dates_dataframe(self.cdr, start_date, end_date)

    def deduplicate(self):

        self.cdr = self.cdr.distinct()


    def get_home_locations(self, algo='count_transactions'):

        if algo == 'count_transactions':
            grouped = self.cdr.groupby(['subscriber_id', self.geo]).agg(count('timestamp').alias('count_transactions'))
            window = Window.partitionBy('subscriber_id').orderBy(desc_nulls_last('count_transactions'))
            grouped = grouped.withColumn('order', row_number().over(window))\
                .where(col('order') == 1)\
                .select(['subscriber_id', self.geo, 'count_transactions'])
        
        elif algo == 'count_days':
            grouped = self.cdr.groupby(['subscriber_id', self.geo]).agg(countDistinct('day').alias('count_days'))
            window = Window.partitionBy('subscriber_id').orderBy(desc_nulls_last('count_days'))
            grouped = grouped.withColumn('order', row_number().over(window))\
                .where(col('order') == 1)\
                .select(['subscriber_id', self.geo, 'count_days'])

        elif algo == 'count_modal_days':
            grouped = self.cdr.groupby(['subscriber_id', 'day', self.geo]).agg(count('timestamp').alias('count_transactions_per_day'))
            window = Window.partitionBy(['subscriber_id', 'day']).orderBy(desc_nulls_last('count_transactions_per_day'))
            grouped = grouped.withColumn('order', row_number().over(window))\
                .where(col('order') == 1)\
                .groupby(['subscriber_id', self.geo])\
                .agg(count('order').alias('count_modal_days'))
            window = Window.partitionBy(['subscriber_id']).orderBy(desc_nulls_last('count_modal_days'))
            grouped = grouped.withColumn('order', row_number().over(window))\
                .where(col('order') == 1)\
                .select(['subscriber_id', self.geo, 'count_modal_days'])

        else:
            raise ValueError('Home location algorithm not recognized. Must be one of count_transactions, count_days, or count_modal_days')

        grouped = grouped.toPandas()
        grouped.to_csv(self.outputs + '/outputs/' + self.geo + '_' + algo + '.csv', index=False)
        self.home_locations[algo] = grouped
        return grouped

    def accuracy(self, algo='count_transactions', table=True, map=True):

        if self.groundtruth is None:
            raise ValueError('Ground truth dataset must be loaded to calculate accuracy statistics.')
        
        # Inner join ground truth data and inferred home locations
        merged = self.home_locations[algo].rename({self.geo: self.geo + '_inferred'}, axis=1)\
            .merge(self.groundtruth.rename({self.geo: self.geo + '_groundtruth'}, axis=1), on='subscriber_id', how='inner')
        print('Observations with inferred home location: %i (%i unique)' % (len(self.home_locations[algo]), \
            len(self.home_locations[algo]['subscriber_id'].unique())))
        print('Observations with ground truth home location: %i (%i unique)' % (len(self.groundtruth), len(self.groundtruth['subscriber_id'].unique())))
        print('Observations with both: %i (%i unique)' % (len(merged), len(merged['subscriber_id'].unique())))

        # Correct observatiosn are ones where the groundtruth home location is the same as the inferred home location
        merged['correct'] = merged[self.geo + '_inferred'] == merged[self.geo + '_groundtruth']

        # Calculate overall accuracy (percent of observations correctly located)
        overall_accuracy = merged['correct'].mean()
        print('Overall accuracy: %.2f' % overall_accuracy)

        # Calculate precision and recall for each antenna/tower/polygon
        recall = merged.rename({self.geo + '_groundtruth': self.geo, 'correct':'recall'}, axis=1)[[self.geo, 'recall']]\
            .groupby(self.geo, as_index=False).agg('mean')
        precision = merged.rename({self.geo + '_inferred': self.geo, 'correct':'precision'}, axis=1)[[self.geo, 'precision']]\
            .groupby(self.geo, as_index=False).agg('mean')
        table = recall.merge(precision, on=self.geo, how='outer').fillna(0).sort_values('precision', ascending=False)
        table['overall_accuracy'] = overall_accuracy

        # Save table
        table.to_csv(self.outputs + '/tables/' + self.geo + '_' + algo + '.csv', index=False)
        self.accuracy_tables[algo] = table
        return table

    def map(self, algo='count_transactions', kind='population', voronoi=False):

        if self.antennas is None:
            raise ValueError('Antennas must be loaded to construct maps.')
        
        if kind not in ['population', 'poverty', 'precision', 'recall']:
            raise ValueError('Map types are population, poverty, precision, and recall')
            
        if kind == 'poverty' and self.poverty_scores is None:
            raise ValueError('Poverty scores must be loaded to construct poverty map.')

        if kind in ['precision', 'recall'] and self.accuracy_tables == {}:
            raise ValueError('Accuracy must be calculated to construct accuracy map.')

        # Get population assigned to each antenna/tower/polygon
        population = self.home_locations[algo].groupby(self.geo).agg('count').rename({'subscriber_id':'population'}, axis=1)
        
        # For poverty map, get average polygon for subscribers assigned to each antenna/tower/polygon and merge with population data
        if kind == 'poverty':
            poverty = self.home_locations[algo].merge(self.poverty_scores.rename({'name':'subscriber_id'}, axis=1), on='subscriber_id', how='inner')
            poverty = poverty.groupby(self.geo).agg('mean').rename({'predicted':'poverty'}, axis=1)
            population = population.merge(poverty, on=self.geo, how='left')
        
        # If accuracy map, merge accuracy data with population data
        elif kind in ['precision', 'recall']:
            population = population.merge(self.accuracy_tables[algo], on=self.geo, how='left')

        if self.geo in ['antenna_id', 'tower_id']:

            # Get pandas dataframes of antennas/towers
            if self.geo == 'antenna_id':
                points = self.antennas.toPandas().dropna(subset=['antenna_id', 'latitude', 'longitude'])
            else:
                points = self.antennas.toPandas()[['tower_id', 'latitude', 'longitude']].dropna().drop_duplicates().copy()

            # Calculate voronoi tesselation and merge to population data
            if voronoi:
                if len(self.shapefiles.keys()) == 0:
                    raise ValueError('At least one shapefile must be loaded to compute voronoi polygons.')
                voronoi_polygons = voronoi_tessellation(points, list(self.shapefiles.values())[0], key=self.geo)
                population = voronoi_polygons.merge(population, on=self.geo, how='left')
            
            # If not voronoi, create geodataframe of latitude/longitude coordinates and merge to population data
            else:
                points = gpd.GeoDataFrame(points, geometry = gpd.points_from_xy(points['longitude'], points['latitude']))
                population = points.merge(population, on=self.geo, how='left')
        
        # If polygons, merge polygon shapefile to population data
        elif self.geo in self.shapefiles.keys():
            population = self.shapefiles[self.geo].rename({'region':self.geo}, axis=1).merge(population, on=self.geo, how='left')

        else:
            raise ValueError('Invalid geometry.')

        # Null population after left join means 0 population assigned to antenna/tower/polygon
        population['population'] = population['population'].fillna(0)

        # Save shapefile
        population.to_file(self.outputs + '/maps/' + self.geo + '_' + algo + '_' + kind + '_voronoi' + str(voronoi) + '.geojson', driver='GeoJSON')

        # Normalize population data for map
        population['population'] = population['population']/population['population'].sum()
        
        # Create map
        fig, ax = plt.subplots(1, figsize=(10, 10))

        if self.geo in ['antenna_id', 'tower_id'] and voronoi==False:
            
            # If points map and shapefile loaded, plot shapefile as background to map
            if len(self.shapefiles.keys()) > 0:
                list(self.shapefiles.values())[0].plot(ax=ax, color='lightgrey')
            
            # Plot points, sized by population and colored by outcome of interest. Plot points with no population/outcome in light grey. 
            population.plot(ax=ax, column=kind, markersize=population['population']*10000, legend=True, legend_kwds={'shrink':0.5})
            population[(pd.isnull(population[kind])) | (population['population'] == 0)].plot(ax=ax, color='grey', markersize=10, legend=False)

        else:

            # Plot polygons, colored by outcome of interest. Plot polygons with no outcome in light grey.
            population.plot(ax=ax, color='lightgrey')
            population.plot(ax=ax, column=kind, legend=True, legend_kwds={'shrink':0.5})

        # Clean and save plot
        ax.axis('off')
        title = 'Population Map' if kind == 'population' else 'Poverty Map' if kind == 'poverty' else 'Precision Map' if kind == 'precision' \
             else 'Recall Map'
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(self.outputs + '/maps/' + self.geo + '_' + algo + '_' + kind + '_voronoi' + str(voronoi) + '.png', dpi=300)
        plt.show()

    def pop_comparison(self, algo='count_transactions'):
        # Get population assigned to each antenna/tower/polygon
        if algo in self.home_locations:
            homes = self.home_locations[algo].groupby(self.geo).agg('count')\
                                                  .rename({'subscriber_id': 'population'}, axis=1).reset_index()
        else:
            raise ValueError(f"Home locations have not been computed for '{algo}' algo")

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

        # Read raster with population data, mask with each shape and compute population in units
        raster_fpath = self.data + self.cfg.path.home_location.file_names.population
        out_data = []
        with rasterio.open(raster_fpath) as src:
            for _, row in shapes.iterrows():
                idx = row[self.geo]
                geometry = row['geometry']
                geoms = [mapping(row['geometry'])]
                out_image, out_transform = mask(src, geoms, crop=True)
                out_image = np.nan_to_num(out_image)
                out_data.append([idx, geometry, out_image.sum()])
        population = pd.DataFrame(data=out_data, columns=['region', 'geometry', 'pop'])

        # Merge with number of locations and compute pct point difference of relative populations
        homes = homes.rename(columns={self.geo: 'region', 'population': 'homes'})
        df = pd.merge(population, homes[['region', 'homes']], on='region', how='left')
        df = df.fillna(0)

        df['pop_pct'] = df['pop'] / df['pop'].sum() * 100
        df['homes_pct'] = df['homes'] / df['homes'].sum() * 100
        df['diff'] = df['homes_pct'] - df['pop_pct']
        df = gpd.GeoDataFrame(df)

        # Save shapefile
        df.to_file(self.outputs + '/maps/' + self.geo + '_' + algo + '_' + 'pop_comparisons' + '.geojson',
                   driver='GeoJSON')

        # Plot map
        fig, ax = plt.subplots(1, figsize=(10, 10))

        df.plot(ax=ax, color='lightgrey')
        df.plot(ax=ax, column='diff', cmap='RdYlGn', legend=True, legend_kwds={'shrink': 0.5})

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.outputs + '/maps/' + self.geo + '_' + algo + '_' + 'pop_comparisons' + '.png', dpi=300)
        plt.show()
