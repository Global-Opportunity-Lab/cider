# Copyright ©2022-2023. The Regents of the University of California
# (Regents). All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the
# distribution.
#
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

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Type

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from geopandas import GeoDataFrame  # type: ignore[import]
from pandas import DataFrame as PandasDataFrame, Series

from cider.datastore import DataStore, DataType, OptDataStore

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("datastore_class", [DataStore, OptDataStore])
class TestDatastoreClasses:
    """Core tests for DataStore implementations using Dask."""

    @pytest.mark.unit_test
    @pytest.mark.parametrize(
        "config_file_path",
        [
            "configs/config_new.yml",
            "configs/config.yml",
        ],
    )
    def test_config_datastore(self, config_file_path: str, datastore_class: Type[DataStore]) -> None:
        datastore_class(config_file_path_string=str(PROJECT_ROOT / config_file_path))

    @pytest.mark.unit_test
    @pytest.mark.parametrize(
        "config_file_path,expected_exception",
        [("", FileNotFoundError), ("\\malformed#$directory!!!(38", FileNotFoundError)],
    )
    def test_config_datastore_exception(
        self,
        config_file_path: str,
        datastore_class: Type[DataStore],
        expected_exception: Type[Exception],
    ) -> None:
        with pytest.raises(expected_exception):
            datastore_class(config_file_path_string=config_file_path)

    @pytest.fixture()
    def ds(self, datastore_class: Type[DataStore]) -> DataStore:
        # Use the existing synthetic_data referenced by this config.
        return datastore_class(config_file_path_string="configs/test_config.yml")

    @pytest.mark.unit_test
    def test_load_cdr_from_disk(self, ds: DataStore) -> None:
        ds._load_cdr()
        assert isinstance(ds.cdr, dd.DataFrame)
        assert len(ds.cdr) == 100000
        assert "day" in ds.cdr.columns

    @pytest.mark.unit_test
    def test_load_cdr_from_df(self, ds: DataStore) -> None:
        test_df = pd.DataFrame(
            data={
                "txn_type": ["text"],
                "caller_id": ["A"],
                "recipient_id": ["B"],
                "timestamp": ["2021-01-01"],
                "duration": [60],
                "international": ["domestic"],
            }
        )
        ds._load_cdr(dataframe=test_df)
        assert isinstance(ds.cdr, dd.DataFrame)
        assert len(ds.cdr) == 1
        assert "day" in ds.cdr.columns

    @pytest.mark.unit_test
    def test_load_antennas_from_disk(self, ds: DataStore) -> None:
        ds._load_antennas()
        assert isinstance(ds.antennas, dd.DataFrame)
        assert len(ds.antennas) == 297
        assert dict(ds.antennas.dtypes)["latitude"] in ("float64", "float32")

    @pytest.mark.unit_test
    def test_load_recharges_from_disk(self, ds: DataStore) -> None:
        ds._load_recharges()
        assert isinstance(ds.recharges, dd.DataFrame)
        assert len(ds.recharges) == 10000
        assert "day" in ds.recharges.columns

    @pytest.mark.unit_test
    def test_load_mobiledata_from_disk(self, ds: DataStore) -> None:
        ds._load_mobiledata()
        assert isinstance(ds.mobiledata, dd.DataFrame)
        assert len(ds.mobiledata) == 10000
        assert "day" in ds.mobiledata.columns

    @pytest.mark.unit_test
    def test_load_mobilemoney_from_disk(self, ds: DataStore) -> None:
        ds._load_mobilemoney()
        assert isinstance(ds.mobilemoney, dd.DataFrame)
        assert len(ds.mobilemoney) == 10000
        assert "day" in ds.mobilemoney.columns

    @pytest.mark.unit_test
    def test_load_shapefiles(self, ds: DataStore) -> None:
        ds._load_shapefiles()
        assert isinstance(ds.shapefiles, dict)
        assert "regions" in ds.shapefiles
        assert isinstance(ds.shapefiles["regions"], GeoDataFrame)
        assert len(ds.shapefiles) == 2

    @pytest.mark.unit_test
    def test_load_home_ground_truth(self, ds: DataStore) -> None:
        ds._load_home_ground_truth()
        assert isinstance(ds.home_ground_truth, PandasDataFrame)
        assert ds.home_ground_truth.shape[0] == 1000

    @pytest.mark.unit_test
    def test_load_features_and_labels_and_merge(self, ds: DataStore) -> None:
        ds._load_features()
        ds._load_labels()
        assert isinstance(ds.features, dd.DataFrame)
        assert isinstance(ds.labels, dd.DataFrame)

        ds.merge()
        assert isinstance(ds.merged, PandasDataFrame)
        assert ds.merged.shape[0] == 50

        assert isinstance(ds.x, PandasDataFrame)
        assert all(col not in ds.x.columns for col in ["name", "label", "weight"])
        assert isinstance(ds.y, Series)
        assert isinstance(ds.weights, Series)
        assert ds.weights.min() >= 1

    @pytest.mark.unit_test
    def test_filter_dates(self, ds: DataStore) -> None:
        ds._load_recharges()
        ds._load_mobiledata()
        min_date, max_date = datetime(2020, 1, 1), datetime(2020, 2, 29)

        # Larger boundaries should keep original range.
        ds.filter_dates(min_date - timedelta(days=1), max_date + timedelta(days=1))
        recharges_pd = ds.recharges[["day"]].compute()
        mobiledata_pd = ds.mobiledata[["day"]].compute()
        assert recharges_pd["day"].min().to_pydatetime() == min_date
        assert recharges_pd["day"].max().to_pydatetime() == max_date
        assert mobiledata_pd["day"].min().to_pydatetime() == min_date
        assert mobiledata_pd["day"].max().to_pydatetime() == max_date

        # Smaller boundaries should shrink range.
        new_min_date, new_max_date = min_date + timedelta(days=1), max_date - timedelta(days=1)
        ds.filter_dates(new_min_date, new_max_date)
        recharges_pd = ds.recharges[["day"]].compute()
        mobiledata_pd = ds.mobiledata[["day"]].compute()
        assert recharges_pd["day"].min().to_pydatetime() == new_min_date
        assert recharges_pd["day"].max().to_pydatetime() == new_max_date
        assert mobiledata_pd["day"].min().to_pydatetime() == new_min_date
        assert mobiledata_pd["day"].max().to_pydatetime() == new_max_date

    @pytest.mark.unit_test
    @pytest.mark.parametrize(
        "df, n_rows",
        [
            (
                pd.DataFrame(
                    data={
                        "caller_id": ["A", "A"],
                        "volume": [50, 50],
                        "timestamp": ["2020-01-01 12:00:00", "2020-01-02 12:00:01"],
                    }
                ),
                2,
            ),
            (
                pd.DataFrame(
                    data={
                        "caller_id": ["A", "A"],
                        "volume": [50, 50],
                        "timestamp": ["2020-01-02 12:00:00", "2020-01-02 12:00:00"],
                    }
                ),
                1,
            ),
        ],
    )
    def test_deduplicate(self, ds: DataStore, df: pd.DataFrame, n_rows: int) -> None:
        ds._load_mobiledata(dataframe=df)
        ds.deduplicate()
        assert len(ds.mobiledata) == n_rows

    @pytest.mark.unit_test
    @pytest.mark.parametrize(
        "df, threshold, n_spammers",
        [
            (
                pd.DataFrame(
                    data={
                        "txn_type": ["call", "call"],
                        "caller_id": ["A", "A"],
                        "recipient_id": ["X", "X"],
                        "duration": [50, 50],
                        "timestamp": ["2020-01-01 12:00:00", "2020-01-02 12:00:01"],
                        "international": ["domestic", "domestic"],
                    }
                ),
                1,
                0,
            ),
            (
                pd.DataFrame(
                    data={
                        "txn_type": ["call"] * 12,
                        "caller_id": ["A", "A"] + ["B"] * 10,
                        "recipient_id": ["X"] * 12,
                        "duration": [50] * 12,
                        "timestamp": ["2020-01-02 12:00:00"] * 12,
                        "international": ["domestic"] * 12,
                    }
                ),
                5,
                1,
            ),
        ],
    )
    def test_remove_spammers(self, ds: DataStore, df: pd.DataFrame, threshold: float, n_spammers: int) -> None:
        ds._load_cdr(dataframe=df)
        spammers = ds.remove_spammers(spammer_threshold=threshold)
        assert len(spammers) == n_spammers
        if spammers:
            remaining = ds.cdr[ds.cdr["caller_id"].isin(spammers)]
            assert len(remaining) == 0

    @pytest.mark.unit_test
    def test_remove_spammers_raises(self, ds: DataStore) -> None:
        ds._load_recharges()
        with pytest.raises(ValueError):
            ds.remove_spammers(spammer_threshold=1)

