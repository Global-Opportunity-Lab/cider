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
import pandas as pd
from cider.schemas import CallDataRecordData, AntennaData
from cider.homelocation.schemas import GeographicUnit, GetHomeLocationAlgorithm
from cider.homelocation.core import (
    _prepare_home_location_data,
    _infer_home_locations,
    get_home_locations,
    get_accuracy,
)
from cider.homelocation.dependencies import (
    _deduplicate_points_within_buffer,
    get_voronoi_tessellation,
)
from conftest import (
    CDR_DATA,
    ANTENNA_DATA,
    SHAPEFILE_DATA,
    HOME_LOCATION_GT,
    POINTS_DATA,
)
import geopandas as gpd


def _get_cdr_data_payload(input: str = "base") -> pd.DataFrame:
    match input:
        case "base":
            return pd.DataFrame(CDR_DATA)
        case "invalid_field":
            cdr_data_invalid = CDR_DATA.copy()
            cdr_data_invalid["invalid_field"] = [1, 2, 3, 4, 5, 6]
            return pd.DataFrame(cdr_data_invalid)
        case "missing_field":
            cdr_data_missing = CDR_DATA.copy()
            cdr_data_missing.pop("duration")
            return pd.DataFrame(cdr_data_missing)
        case "invalid_transaction_type":
            cdr_data_invalid_type = CDR_DATA.copy()
            cdr_data_invalid_type["transaction_type"] = ["text", "invalid"] * 3
            return pd.DataFrame(cdr_data_invalid_type)
        case "invalid_transaction_scope":
            cdr_data_invalid_scope = CDR_DATA.copy()
            cdr_data_invalid_scope["transaction_scope"] = ["domestic", "invalid"] * 3
            return pd.DataFrame(cdr_data_invalid_scope)


def _get_antenna_data_payload(input: str = "base") -> pd.DataFrame:
    match input:
        case "base":
            return pd.DataFrame(ANTENNA_DATA)
        case "invalid_field":
            antenna_data_invalid = ANTENNA_DATA.copy()
            antenna_data_invalid["invalid_field"] = [1, 2, 3]
            return pd.DataFrame(antenna_data_invalid)
        case "missing_field":
            antenna_data_missing = ANTENNA_DATA.copy()
            antenna_data_missing.pop("latitude")
            return pd.DataFrame(antenna_data_missing)
        case "missing_tower_id":
            antenna_data_missing_tower = ANTENNA_DATA.copy()
            antenna_data_missing_tower.pop("tower_id")
            return pd.DataFrame(antenna_data_missing_tower)
        case "renamed_tower_id":
            antenna_data_renamed = ANTENNA_DATA.copy()
            antenna_data_renamed["tower_id"] = ["tower_1", "tower_2", "tower_3"]
            return pd.DataFrame(antenna_data_renamed)


class TestHomeLocationCore:
    @pytest.fixture
    def create_cdr_data_schema(self, request: pytest.FixtureRequest) -> dict:
        """Fixture to create experiment payload based on request parameter."""
        return _get_cdr_data_payload(request.param)

    @pytest.fixture
    def create_antenna_data_schema(self, request: pytest.FixtureRequest) -> dict:
        """Fixture to create experiment payload based on request parameter."""
        return _get_antenna_data_payload(request.param)

    @pytest.mark.parametrize(
        "create_cdr_data_schema",
        [
            "invalid_field",
            "missing_field",
            "invalid_transaction_type",
            "invalid_transaction_scope",
        ],
        indirect=["create_cdr_data_schema"],
    )
    def test_cdr_data_validation(self, create_cdr_data_schema):
        with pytest.raises(ValueError):
            CallDataRecordData.model_validate(create_cdr_data_schema)

    @pytest.mark.parametrize(
        "create_antenna_data_schema",
        ["invalid_field", "missing_field"],
        indirect=["create_antenna_data_schema"],
    )
    def test_antenna_data_validation(self, create_antenna_data_schema):
        with pytest.raises(ValueError):
            AntennaData.model_validate(create_antenna_data_schema)

    @pytest.mark.parametrize(
        "create_cdr_data_schema,create_antenna_data_schema,geographic_unit",
        [
            ("base", "base", GeographicUnit.ANTENNA_ID),
            ("base", "base", GeographicUnit.TOWER_ID),
            ("base", "base", GeographicUnit.SHAPEFILE),
        ],
        indirect=["create_cdr_data_schema", "create_antenna_data_schema"],
    )
    def test_prepare_home_location_data(
        self, create_cdr_data_schema, create_antenna_data_schema, geographic_unit
    ):
        cdr_df = create_cdr_data_schema
        antenna_df = create_antenna_data_schema

        prepared_data = _prepare_home_location_data(
            cdr_data=cdr_df,
            antenna_data=antenna_df,
            geographic_unit=geographic_unit,
            shapefile_data=(
                SHAPEFILE_DATA if geographic_unit == GeographicUnit.SHAPEFILE else None
            ),
        )
        if geographic_unit == GeographicUnit.SHAPEFILE:
            prepared_data.drop(columns=["region", "index_right"], inplace=True)
        assert not prepared_data.empty
        assert prepared_data.shape == (6, 11)
        assert set(prepared_data.columns).issubset(
            set(cdr_df.columns).union(set(antenna_df.columns)) - {geographic_unit}
        )

    @pytest.mark.parametrize(
        "create_cdr_data_schema,create_antenna_data_schema,geographic_unit",
        [
            ("base", "renamed_tower_id", GeographicUnit.TOWER_ID),
            ("base", "missing_tower_id", GeographicUnit.TOWER_ID),
            ("base", "base", "incorrect_geographic_unit"),
            ("base", "base", GeographicUnit.SHAPEFILE),
        ],
        indirect=["create_cdr_data_schema", "create_antenna_data_schema"],
    )
    def test_prepare_home_location_data_invalid_data(
        self, create_cdr_data_schema, create_antenna_data_schema, geographic_unit
    ):
        cdr_df = create_cdr_data_schema
        antenna_df = create_antenna_data_schema

        with pytest.raises(ValueError):
            _prepare_home_location_data(
                cdr_data=cdr_df,
                antenna_data=antenna_df,
                geographic_unit=geographic_unit,
            )

    @pytest.mark.parametrize(
        "create_cdr_data_schema,create_antenna_data_schema,algorithm",
        [
            (
                "base",
                "base",
                GetHomeLocationAlgorithm.COUNT_TRANSACTIONS,
            ),
            ("base", "base", GetHomeLocationAlgorithm.COUNT_DAYS),
            ("base", "base", GetHomeLocationAlgorithm.COUNT_MODAL_DAYS),
        ],
        indirect=["create_cdr_data_schema", "create_antenna_data_schema"],
    )
    def test_core_home_location(
        self,
        create_cdr_data_schema,
        create_antenna_data_schema,
        spark,
        algorithm,
    ):
        cdr_df = create_cdr_data_schema
        antenna_df = create_antenna_data_schema

        prepared_data = _prepare_home_location_data(
            cdr_data=cdr_df,
            antenna_data=antenna_df,
            geographic_unit=GeographicUnit.ANTENNA_ID,
        )

        # Infer home locations
        inferred_locations = _infer_home_locations(
            prepared_data=prepared_data,
            algorithm=algorithm,
            spark_session=spark,
        )
        assert not inferred_locations.empty
        assert inferred_locations.shape == (3, 3)
        assert set(inferred_locations.columns) == {
            "caller_id",
            "caller_antenna_id",
            algorithm.value,
        }
        assert inferred_locations[algorithm.value].tolist() == [1, 1, 1]

        # Infer home locations with additional columns to keep
        inferred_locations = _infer_home_locations(
            prepared_data=prepared_data,
            algorithm=algorithm,
            spark_session=spark,
            additional_columns_to_keep=["transaction_type"],
        )
        assert not inferred_locations.empty
        assert inferred_locations.shape == (3, 4)
        assert set(inferred_locations.columns) == {
            "caller_id",
            "caller_antenna_id",
            algorithm.value,
            "transaction_type",
        }
        assert inferred_locations[algorithm.value].tolist() == [1, 1, 1]

        with pytest.raises(ValueError):
            inferred_locations = _infer_home_locations(
                prepared_data=prepared_data,
                algorithm=algorithm,
                spark_session=spark,
                additional_columns_to_keep=["some_invalid_column"],
            )

    @pytest.mark.parametrize(
        "create_cdr_data_schema,create_antenna_data_schema,column_to_merge_on,column_to_measure_on",
        [
            ("base", "base", "caller_id", "caller_antenna_id"),
            ("base", "base", "caller_id", "region"),
            ("base", "base", "caller_antenna_id", "region"),
        ],
        indirect=["create_cdr_data_schema", "create_antenna_data_schema"],
    )
    def test_home_location_core_accuracy(
        self,
        spark,
        create_cdr_data_schema,
        create_antenna_data_schema,
        column_to_merge_on,
        column_to_measure_on,
    ):
        cdr_data = create_cdr_data_schema
        antenna_data = create_antenna_data_schema
        home_locations = get_home_locations(
            cdr_data=cdr_data,
            antenna_data=antenna_data,
            geographic_unit=GeographicUnit.SHAPEFILE,
            algorithm=GetHomeLocationAlgorithm.COUNT_DAYS,
            shapefile_data=SHAPEFILE_DATA,
            spark_session=spark,
            additional_columns_to_keep=["region"],
        )
        accuracy_table = get_accuracy(
            inferred_home_locations=home_locations,
            groundtruth_home_locations=HOME_LOCATION_GT,
            column_to_merge_on=column_to_merge_on,
            column_to_measure_on=column_to_measure_on,
        )
        assert not accuracy_table.empty
        assert set(accuracy_table.columns) == {
            column_to_measure_on + "_groundtruth",
            column_to_measure_on + "_inferred",
        } | {"recall", "precision"}
        assert accuracy_table.shape[0] >= 1

    @pytest.mark.parametrize(
        "create_cdr_data_schema,create_antenna_data_schema,column_to_merge_on,column_to_measure_on",
        [
            ("base", "base", "some_invalid_id", "caller_antenna_id"),
            ("base", "base", "caller_id", "some_invalid_id"),
        ],
        indirect=["create_cdr_data_schema", "create_antenna_data_schema"],
    )
    def test_home_location_core_accuracy_failure(
        self,
        spark,
        create_cdr_data_schema,
        create_antenna_data_schema,
        column_to_merge_on,
        column_to_measure_on,
    ):
        cdr_data = create_cdr_data_schema
        antenna_data = create_antenna_data_schema
        home_locations = get_home_locations(
            cdr_data=cdr_data,
            antenna_data=antenna_data,
            geographic_unit=GeographicUnit.SHAPEFILE,
            algorithm=GetHomeLocationAlgorithm.COUNT_DAYS,
            shapefile_data=SHAPEFILE_DATA,
            spark_session=spark,
            additional_columns_to_keep=["region"],
        )
        with pytest.raises(ValueError):
            get_accuracy(
                inferred_home_locations=home_locations,
                groundtruth_home_locations=HOME_LOCATION_GT,
                column_to_merge_on=column_to_merge_on,
                column_to_measure_on=column_to_measure_on,
            )


class TestHomeLocationDependencies:

    def test_deduplication_code(self):
        deduplicated_points = _deduplicate_points_within_buffer(
            xy_points=POINTS_DATA, points_id_col="ids", buffer_distance=1e4
        )
        assert not deduplicated_points.empty
        assert len(deduplicated_points) == 2
        assert set(deduplicated_points["ids"]) == {"a", "b"}

    def test_voronoi_tesellation_code(self):
        antenna_data = pd.DataFrame(ANTENNA_DATA)
        antenna_data = gpd.GeoDataFrame(
            antenna_data,
            geometry=gpd.points_from_xy(
                antenna_data["longitude"], antenna_data["latitude"]
            ),
            crs="EPSG:4326",
        )
        voronoi_gdf = get_voronoi_tessellation(
            xy_points=antenna_data,
            boundary_shapefile=SHAPEFILE_DATA,
            points_id_col="antenna_id",
            buffer_distance_for_deduplication=1e-6,
        )
        assert not voronoi_gdf.empty
        assert len(voronoi_gdf) == 3
        assert set(voronoi_gdf["antenna_id"]) == {"antenna_1", "antenna_2", "antenna_3"}
