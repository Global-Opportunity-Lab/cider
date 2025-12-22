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


import pytest
from pyspark.sql import SparkSession
import geopandas as gpd
import pandas as pd

from shapely import Polygon


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master("local[1]").appName("pytest-spark").getOrCreate()
    )
    yield spark
    spark.stop()


CDR_DATA = {
    "caller_id": ["caller_1"] * 2 + ["caller_2"] * 2 + ["caller_3"] * 2,
    "recipient_id": ["recipient_1"] * 6,
    "caller_antenna_id": ["antenna_1", "antenna_2"] * 3,
    "recipient_antenna_id": ["antenna_3", "antenna_4"] * 3,
    "timestamp": pd.to_datetime(
        [
            "2023-01-01 10:00:00",
            "2023-01-02 12:00:00",
            "2023-01-02 14:00:00",
            "2023-01-04 16:00:00",
            "2023-01-05 18:00:00",
            "2023-01-06 21:00:00",
        ]
    ),
    "duration": [300, 200, 400, 100, 250, 150],
    "transaction_type": ["text", "call"] * 3,
    "transaction_scope": ["domestic"] * 2 + ["international"] * 2 + ["other"] * 2,
}


ANTENNA_DATA = {
    "antenna_id": ["antenna_1", "antenna_2", "antenna_3"],
    "tower_id": ["antenna_1", "antenna_2", "antenna_3"],
    "latitude": [1.5001, 2.4987, 3.3467],
    "longitude": [1.8965, 2.4231, 3.0078],
}


SHAPEFILE_DATA = gpd.GeoDataFrame(
    {"region": ["region_1", "region_2", "region_3"]},
    geometry=[
        Polygon(
            [(1.1920, 1.1245), (4.4358, 1.2395), (4.3526, 4.9873), (1.1557, 4.7873)]
        ),
        Polygon(
            [(4.3467, 4.8236), (4.7957, 6.2368), (6.5823, 6.2366), (6.6757, 4.1905)]
        ),
        Polygon(
            [(0.5873, 3.6922), (0.2684, 0.1578), (3.6124, 3.3649), (3.9823, 0.2396)]
        ),
    ],
)

SHAPEFILE_DATA.set_crs("EPSG:4326", inplace=True)
SHAPEFILE_DATA["geometry"] = SHAPEFILE_DATA.buffer(0)

HOME_LOCATION_GT = pd.DataFrame(
    {
        "caller_id": ["caller_1", "caller_2", "caller_3"],
        "caller_antenna_id": ["antenna_1", "antenna_1", "antenna_2"],
        "region": ["region_1", "region_1", "region_2"],
    }
)


POINTS_DATA = gpd.GeoDataFrame(
    {
        "ids": ["a", "b", "c"],
        "geometry": gpd.points_from_xy(
            [0.0001, 0.1, 0.00003], [0.0004, 0.0004, 0.0004]
        ),
    }
)
POINTS_DATA = POINTS_DATA.set_crs(epsg=4326)
POINTS_DATA = POINTS_DATA.to_crs(epsg=3857)


RECHARGE_DATA = {
    "caller_id": ["caller_1", "caller_2", "caller_3"] * 2,
    "timestamp": pd.to_datetime(
        [
            "2023-01-01 9:00:00",
            "2023-01-01 11:30:00",
            "2023-01-02 08:00:00",
            "2023-01-02 12:20:00",
            "2023-01-03 07:30:00",
            "2023-01-03 11:00:00",
        ]
    ),
    "amount": [100.0, 150.0, 200.0] * 2,
}

MOBILE_DATA_USAGE_DATA = {
    "caller_id": ["caller_1", "caller_2", "caller_3"] * 2,
    "timestamp": pd.to_datetime(
        [
            "2023-01-01 07:10:00",
            "2023-01-01 09:40:00",
            "2023-01-02 11:56:00",
            "2023-01-02 12:01:00",
            "2023-01-03 08:45:00",
            "2023-01-03 12:32:00",
        ]
    ),
    "volume": [500.0, 750.0, 1000.0] * 2,
}

MOBILE_MONEY_TRANSACTION_DATA = {
    "caller_id": ["caller_1", "caller_2", "caller_3"] * 2,
    "recipient_id": ["recipient_1", "recipient_2", "recipient_3"] * 2,
    "timestamp": pd.to_datetime(
        [
            "2023-01-01 05:36:00",
            "2023-01-01 06:12:00",
            "2023-01-02 03:56:00",
            "2023-01-02 10:29:00",
            "2023-01-03 08:44:00",
            "2023-01-03 12:00:00",
        ]
    ),
    "transaction_type": ["cashin", "cashout", "other", "p2p", "billpay", "other"],
    "amount": [1000.0, 1500.0, 2000.0] * 2,
    "caller_balance_before": [5000.0, 6000.0, 7000.0] * 2,
    "caller_balance_after": [4000.0, 4500.0, 5000.0] * 2,
    "recipient_balance_before": [2000.0, 2500.0, 3000.0] * 2,
    "recipient_balance_after": [3000.0, 4000.0, 5000.0] * 2,
}
