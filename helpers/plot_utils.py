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
import dask.dataframe as dd
import seaborn as sns  # type: ignore[import]
from typing import List, Union

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


def distributions_plot(
    df: Union[dd.DataFrame, PandasDataFrame],
    features: List[str],
    names: List[str],
    color: str = 'indianred',
) -> None:
    """
    Plot distribution of computed features, using Kernel Density Estimation

    Args:
        df: dask or pandas df with features
        features: list of feature names to plot
        names: plot titles
        color: color palette to use
    """
    if not isinstance(df, (dd.DataFrame, PandasDataFrame)):
        raise TypeError("distributions_plot expects a dask.dataframe.DataFrame or pandas.DataFrame")

    fig, ax = plt.subplots(1, len(features), figsize=(20, 5))
    for a in range(len(features)):
        if isinstance(df, dd.DataFrame):
            values = df[features[a]].dropna().compute().to_numpy()
        else:
            values = df[features[a]].dropna().to_numpy()

        # seaborn kdeplot errors on empty input; make this failure explicit
        if len(values) == 0:
            raise ValueError(f"Cannot plot KDE: feature '{features[a]}' has no non-null values.")

        sns.kdeplot(values, ax=ax[a], fill=True, color=color)
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

    # geovoronoi returns two dictionaries. Both have arbitrary indices as keys.
    # One maps these to regions, the other to one or more indices into the list 
    # of labels, representing towers in that given region. In theory we have
    # de-duped so there should only be one tower per region, but in practice
    # towers can vary in location by tiny amounts that escape the de-duping but
    # still result in sharing a region. In those cases, we arbitrarily associate
    # one of the towers with the region.
    regions, label_indices = geovoronoi.voronoi_regions_from_coords(coords, shapefile)

    ordered_regions = []
    ordered_labels = []

    for i in regions.keys():
        ordered_regions.append(regions[i])
        ordered_labels.append(labels[label_indices[i][0]])

    voronoi = PandasDataFrame(data=[ordered_regions, ordered_labels]).T

    voronoi.columns = ['geometry', key]
    voronoi = gpd.GeoDataFrame(voronoi, geometry='geometry')
    voronoi['geometry'] = voronoi['geometry'].convex_hull

    return voronoi
