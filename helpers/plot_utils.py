from helpers.utils import *

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import geopandas as gpd
import geovoronoi

sns.set(font_scale=2, style='white')

# Source: https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib
def clean_plot(ax):

    plt.tight_layout()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def dates_xaxis(ax, frequency):

    if frequency == 'day':
        locator = mdates.DayLocator()
        format = mdates.DateFormatter('%y-%m-%d')  

    elif frequency == 'week':
        locator = mdates.WeekdayLocator()  
        format = mdates.DateFormatter('%y-%m-%d')

    elif frequency == 'month':
        locator = mdates.MonthLocator()  
        format = mdates.DateFormatter('%y-%m')

    elif frequency == 'year':
        locator = mdates.YearLocator()
        format = mdates.DateFormatter('%Y')

    else:
        raise ValueError('Invalid frequency for date axis.')

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(format)

def distributions_plot(df, features, names, color='indianred'):
    
    fig, ax = plt.subplots(1, len(features), figsize=(20, 5))
    for a in range(len(features)):
        sns.kdeplot(df.select(features[a]).rdd.map(lambda r: r[0]).collect(), ax=ax[a], shade=True, color=color)
        if a == 0:
            ax[a].set_ylabel('Density')
        ax[a].set_title(names[a])
        clean_plot(ax[a])
    plt.tight_layout()

def voronoi_tessellation(points, shapefile, key='tower_id'):
    
    points = points[[key, 'latitude', 'longitude']].drop_duplicates().dropna()
    if not len(points[['latitude', 'longitude']].drop_duplicates()) == len(points[key]):
        raise ValueError('Latitude/longitude coordinates must be unique')
    
    coords = points[['longitude', 'latitude']].values
    labels = points[key].values

    shapefile['nation'] = 1
    shapefile = shapefile.dissolve(by='nation')['geometry'].values[0]

    voronoi = geovoronoi.voronoi_regions_from_coords(coords, shapefile)
    voronoi = gpd.GeoDataFrame([list(voronoi[0].values()), [labels[i] for i in flatten_lst(list(voronoi[1].values()))]]).T
    voronoi.columns = ['geometry', key]
    voronoi['geometry'] = voronoi['geometry'].convex_hull
    return voronoi
