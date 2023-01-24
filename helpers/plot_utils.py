# Copyright ©2022-2023. The Regents of the University of California (Regents). 
# All Rights Reserved. 

# Permission to use, copy, modify, and distribute this software and its 
# documentation for educational, research, and not-for-profit purposes, without
# fee and without a signed licensing agreement, is hereby granted, provided that 
# the above copyright notice, this paragraph and the following two paragraphs 
# appear in all copies, modifications, and distributions. Contact The Office of
# Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, 
# CA 94720-1620, (510) 643-7201, otl@berkeley.edu, 
# http://ipira.berkeley.edu/industry-info for commercial licensing 
# opportunities.

# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, 
# SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING
# OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS 
# BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED 
# HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE 
# MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

import geopandas as gpd  # type: ignore[import]
from geopandas import GeoDataFrame
import geovoronoi  # type: ignore[import]
from helpers.utils import flatten_lst
import matplotlib.pyplot as plt  # type: ignore[import]
import matplotlib.dates as mdates  # type: ignore[import]
from matplotlib.pyplot import axis
import matplotlib.ticker as mtick   # type: ignore[import]
from matplotlib.collections import PatchCollection    # type: ignore[import]
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
import seaborn as sns  # type: ignore[import]
from typing import List

sns.set(font_scale=2, style='white')


# Source: https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib
def clean_plot(ax: axis) -> None:
    # Format plot on given axis
    plt.tight_layout()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def dates_xaxis(ax: axis, frequency: str) -> None:
    """
    Format datetime x axis using provided frequency
    
    Args:
        ax: axis to format
        frequency: can be part of ['day', 'week', 'month', 'year']
    """
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


def distributions_plot(df: SparkDataFrame, features: List[str], names: List[str], color: str = 'indianred') -> None:
    """
    Plot distribution of computed features, using Kernel Density Estimation

    Args:
        df: spark df with features
        features: list of feature names to plot
        names: plot titles
        color: color palette to use
    """
    fig, ax = plt.subplots(1, len(features), figsize=(20, 5))
    for a in range(len(features)):
        sns.kdeplot(df.select(features[a]).rdd.map(lambda r: r[0]).collect(), ax=ax[a], shade=True, color=color)
        if a == 0:
            ax[a].set_ylabel('Density')
        ax[a].set_title(names[a])
        clean_plot(ax[a])
    plt.tight_layout()


def voronoi_tessellation(points: PandasDataFrame, shapefile: GeoDataFrame, key: str = 'tower_id') -> GeoDataFrame:
    """
    Create voronoi tessellation starting from points - usually antennas - and a shapefile to define country boundaries

    Args:
        points: pandas df with point geometry column
        shapefile: geopandas df with external boundaries
        key: point identifier

    Returns: geopandas df with geometry column containing voronoi tessellation polygons

    """
    points = points[[key, 'latitude', 'longitude']].drop_duplicates().dropna()
    if not len(points[['latitude', 'longitude']].drop_duplicates()) == len(points[key]):
        raise ValueError('Latitude/longitude coordinates must be unique')
    
    coords = points[['longitude', 'latitude']].values
    labels = points[key].values

    shapefile['nation'] = 1
    shapefile = shapefile.dissolve(by='nation')['geometry'].values[0]

    voronoi = geovoronoi.voronoi_regions_from_coords(coords, shapefile)
    voronoi = PandasDataFrame([list(voronoi[0].values()),
                               [labels[i] for i in flatten_lst(list(voronoi[1].values()))]]).T
    voronoi.columns = ['geometry', key]
    voronoi = gpd.GeoDataFrame(voronoi, geometry='geometry')
    voronoi['geometry'] = voronoi['geometry'].convex_hull

    return voronoi
