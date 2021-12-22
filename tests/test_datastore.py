import os

from datetime import datetime, timedelta
import geopandas
from geopandas import GeoDataFrame  # type: ignore[import]
import numpy as np
import pandas as pd
from pandas import DataFrame as PandasDataFrame, Series
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.utils import AnalysisException
from typing import Dict, MutableMapping, Type

import pytest
from pytest_mock import mocker, MockerFixture

from cider.datastore import DataStore, DataType, OptDataStore
from helpers.utils import get_project_root, get_spark_session

malformed_dataframes_and_errors = {
    'cdr': [(pd.DataFrame(
        data={'txn_type': ['text'], 'caller_id': ['A'], 'recipient_id': ['B'], 'timestamp': ['2021-01-01']}),
             ValueError),
        (pd.DataFrame(data={'txn_type': ['text_message'], 'caller_id': ['A'], 'recipient_id': ['B'],
                            'timestamp': ['2021-01-01'], 'duration': [60], 'international': ['domestic']}),
         ValueError)],
    'antennas': [(pd.DataFrame(data={'antenna_id': ['1'], 'latitude': ['10']}), ValueError)],
    'recharges': [(pd.DataFrame(data={'caller_id': ['A'], 'amount': ['100']}), AnalysisException),
                  (pd.DataFrame(data={'caller_id': ['A'], 'timestamp': ['2020-01-01']}), AnalysisException)],
    'mobiledata': [(pd.DataFrame(data={'caller_id': ['A'], 'timestamp': ['2021-01-01']}), AnalysisException),
                   (pd.DataFrame(data={'caller_id': ['A'], 'volume': ['100']}), AnalysisException)],
    'mobilemoney': [(pd.DataFrame(data={'txn_type': ['cashin'], 'caller_id': ['A'], 'recipient_id': ['B'],
                                        'timestamp': ['2021-01-01']}), ValueError),
                    (pd.DataFrame(data={'txn_type': ['cash-in'], 'caller_id': ['A'], 'recipient_id': ['B'],
                                        'timestamp': ['2021-01-01'], 'amount': [10]}), ValueError)],
    'shapefiles': [(pd.DataFrame(data={'region': ['X']}), ValueError),
                   (pd.DataFrame(data={'geometry': ['A']}), ValueError),
                   (pd.DataFrame(data={'region': ['X'], 'geometry': ['A']}), AssertionError)],
    'labels': [(pd.DataFrame(data={'name': ['A']}), ValueError),
               (pd.DataFrame(data={'label': ['50']}), ValueError)]
}


@pytest.mark.parametrize("datastore_class", [DataStore, OptDataStore])
class TestDatastoreClasses:
    """All the tests related to objects that implement Datastore."""

    @pytest.mark.unit_test
    @pytest.mark.parametrize(
        "config_file_path",
        [
            "configs/config_new.yml",
            "configs/config.yml",
        ],
    )
    def test_config_datastore(
            self, config_file_path: str, datastore_class: Type[DataStore]
    ) -> None:
        """Test that each config file is not stale and can initialize without raising an error."""
        datastore = datastore_class(
            cfg_dir=os.path.join(get_project_root(), config_file_path)
        )

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
            datastore = datastore_class(cfg_dir=config_file_path)

    @pytest.fixture()
    def mock_dataframe_reader(self, mocker: MockerFixture):
        mock_dataframe_reader = mocker.patch("pyspark.sql.session.DataFrameReader", autospec=True)
        return mock_dataframe_reader

    @pytest.fixture()
    def ds(self, datastore_class: Type[DataStore]) -> DataStore:
        out = datastore_class(cfg_dir="configs/test_config.yml")
        return out

    @pytest.mark.unit_test
    def test_load_cdr(self, ds: Type[DataStore]) -> None:  # ds_mock_spark: DataStore
        ds._load_cdr()
        assert isinstance(ds.cdr, SparkDataFrame)
        assert ds.cdr.count() == 1e5
        assert 'day' in ds.cdr.columns
        assert len(ds.cdr.columns) == 9

        test_df = pd.DataFrame(data={'txn_type': ['text'], 'caller_id': ['A'], 'recipient_id': ['B'],
                                     'timestamp': ['2021-01-01'], 'duration': [60], 'international': ['domestic']})
        ds._load_cdr(dataframe=test_df)
        assert isinstance(ds.cdr, SparkDataFrame)
        assert ds.cdr.count() == 1
        assert 'day' in ds.cdr.columns
        assert len(ds.cdr.columns) == 7

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['cdr'])
    def test_load_cdr_raises_from_csv(self, mocker: MockerFixture, ds: DataStore, dataframe, expected_error):
        mock_spark = mocker.patch("helpers.utils.SparkSession", autospec=True)
        mock_read_csv = mock_spark.return_value.read.csv
        mock_read_csv.return_value = ds.spark.createDataFrame(dataframe)
        with pytest.raises(expected_error):
            ds._load_cdr()

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['cdr'])
    def test_load_cdr_raises_from_df(self, ds: DataStore, dataframe, expected_error):
        with pytest.raises(expected_error):
            ds._load_cdr(dataframe=dataframe)

    @pytest.mark.unit_test
    def test_load_antennas(self, ds: Type[DataStore]) -> None:
        ds._load_antennas()
        assert isinstance(ds.antennas, SparkDataFrame)
        assert ds.antennas.count() == 297
        assert dict(ds.antennas.dtypes)['latitude'] == 'float'
        assert len(ds.antennas.columns) == 4

        test_df = pd.DataFrame(data={'antenna_id': ['1'], 'latitude': ['10'], 'longitude': ['25.3']})
        ds._load_antennas(dataframe=test_df)
        assert isinstance(ds.antennas, SparkDataFrame)
        assert ds.antennas.count() == 1
        assert dict(ds.antennas.dtypes)['latitude'] == 'float'
        assert len(ds.antennas.columns) == 3

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['antennas'])
    def test_load_antennas_raises_from_csv(self, mocker: MockerFixture, ds: DataStore, dataframe, expected_error):
        mock_spark = mocker.patch("helpers.utils.SparkSession", autospec=True)
        mock_read_csv = mock_spark.return_value.read.csv
        mock_read_csv.return_value = ds.spark.createDataFrame(dataframe)
        with pytest.raises(expected_error):
            ds._load_antennas()

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['antennas'])
    def test_load_antennas_raises_from_df(self, ds: DataStore, dataframe, expected_error):
        with pytest.raises(expected_error):
            ds._load_antennas(dataframe=dataframe)

    @pytest.mark.unit_test
    def test_load_recharges(self, ds: Type[DataStore]) -> None:
        ds._load_recharges()
        assert isinstance(ds.recharges, SparkDataFrame)
        assert ds.recharges.count() == 1e4
        assert len(ds.recharges.columns) == 4

        test_df = pd.DataFrame(data={'caller_id': ['A'], 'amount': ['100'], 'timestamp': ['2020-01-01']})
        ds._load_recharges(dataframe=test_df)
        assert isinstance(ds.recharges, SparkDataFrame)
        assert ds.recharges.count() == 1
        assert len(ds.recharges.columns) == 4

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['recharges'])
    def test_load_recharges_raises_from_csv(self, mock_dataframe_reader, ds: DataStore, dataframe, expected_error):
        mock_dataframe_reader.return_value.csv.return_value = ds.spark.createDataFrame(dataframe)
        with pytest.raises(expected_error):
            ds._load_recharges()

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['recharges'])
    def test_load_recharges_raises_from_df(self, ds: DataStore, dataframe, expected_error):
        with pytest.raises(expected_error):
            ds._load_recharges(dataframe=dataframe)

    @pytest.mark.unit_test
    def test_load_mobiledata(self, ds: Type[DataStore]) -> None:
        ds._load_mobiledata()
        assert isinstance(ds.mobiledata, SparkDataFrame)
        assert ds.mobiledata.count() == 1e4
        assert len(ds.mobiledata.columns) == 4

        test_df = pd.DataFrame(data={'caller_id': ['A'], 'volume': ['100'], 'timestamp': ['2020-01-01']})
        ds._load_mobiledata(dataframe=test_df)
        assert isinstance(ds.mobiledata, SparkDataFrame)
        assert ds.mobiledata.count() == 1
        assert len(ds.mobiledata.columns) == 4

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['mobiledata'])
    def test_load_mobiledata_raises_from_csv(self, mock_dataframe_reader: MockerFixture, ds: DataStore, dataframe,
                                             expected_error):
        mock_dataframe_reader.return_value.csv.return_value = ds.spark.createDataFrame(dataframe)
        with pytest.raises(expected_error):
            ds._load_mobiledata()

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['mobiledata'])
    def test_load_mobiledata_raises_from_df(self, ds: DataStore, dataframe, expected_error):
        with pytest.raises(expected_error):
            ds._load_mobiledata(dataframe=dataframe)

    @pytest.mark.unit_test
    def test_load_mobilemoney(self, ds: Type[DataStore]) -> None:
        ds._load_mobilemoney()
        assert isinstance(ds.mobilemoney, SparkDataFrame)
        assert ds.mobilemoney.count() == 1e4
        assert len(ds.mobilemoney.columns) == 10

        test_df = pd.DataFrame(data={'txn_type': ['cashin'], 'caller_id': ['A'], 'recipient_id': ['B'],
                                     'timestamp': ['2021-01-01'], 'amount': [10]})
        ds._load_mobilemoney(dataframe=test_df)
        assert isinstance(ds.mobilemoney, SparkDataFrame)
        assert ds.mobilemoney.count() == 1
        assert len(ds.mobilemoney.columns) == 6

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['mobilemoney'])
    def test_load_mobilemoney_raises_from_csv(self, mock_dataframe_reader: MockerFixture, ds: DataStore, dataframe,
                                              expected_error):
        mock_dataframe_reader.return_value.csv.return_value = ds.spark.createDataFrame(dataframe)
        with pytest.raises(expected_error):
            ds._load_mobilemoney()

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['mobilemoney'])
    def test_load_mobilemoney_raises_from_df(self, ds: DataStore, dataframe, expected_error):
        with pytest.raises(expected_error):
            ds._load_mobilemoney(dataframe=dataframe)

    @pytest.mark.unit_test
    def test_load_shapefiles(self, ds: Type[DataStore]) -> None:
        ds._load_shapefiles()
        assert isinstance(ds.shapefiles, dict)
        assert 'regions' in ds.shapefiles
        assert isinstance(ds.shapefiles['regions'], GeoDataFrame)
        assert len(ds.shapefiles) == 2

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['shapefiles'])
    def test_load_shapefiles_raises_from_csv(self, mocker: MockerFixture, ds: Type[DataStore], dataframe,
                                             expected_error) -> None:
        mock_geodataframe_reader = mocker.patch("helpers.io_utils.gpd.read_file", autospec=True)
        mock_geodataframe_reader.return_value = GeoDataFrame(dataframe)
        with pytest.raises(expected_error):
            ds._load_shapefiles()

    @pytest.mark.unit_test
    def test_load_home_ground_truth(self, ds: Type[DataStore]) -> None:
        ds._load_home_ground_truth()
        assert isinstance(ds.home_ground_truth, PandasDataFrame)
        assert ds.home_ground_truth.shape[0] == 1e3

    @pytest.mark.unit_test
    def test_load_poverty_scores(self, ds: Type[DataStore]) -> None:
        # TODO: Create poverty scores fake data
        ds._load_poverty_scores()
        assert isinstance(ds.poverty_scores, PandasDataFrame)

    @pytest.mark.unit_test
    def test_load_features(self, ds: Type[DataStore]) -> None:
        ds._load_features()
        assert isinstance(ds.features, SparkDataFrame)
        assert ds.features.count() == 1e3

    @pytest.mark.unit_test
    def test_load_features_raises(self, mock_dataframe_reader: MockerFixture, ds: Type[DataStore]) -> None:
        dataframe = pd.DataFrame(data={'user_id': ['X'], 'feat0': [50]})
        mock_dataframe_reader.return_value.csv.return_value = ds.spark.createDataFrame(dataframe)
        with pytest.raises(ValueError):
            ds._load_features()

    @pytest.mark.unit_test
    def test_load_labels(self, ds: Type[DataStore]) -> None:
        ds._load_labels()
        assert isinstance(ds.labels, SparkDataFrame)
        assert ds.labels.count() == 50
        assert len(ds.labels.columns) == 3

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['labels'])
    def test_load_labels_raises_from_csv(self, mock_dataframe_reader, ds: Type[DataStore], dataframe,
                                         expected_error) -> None:
        mock_dataframe_reader.return_value.csv.return_value = ds.spark.createDataFrame(dataframe)
        with pytest.raises(expected_error):
            ds._load_labels()

    @pytest.mark.unit_test
    def test_load_targeting(self, ds: Type[DataStore]) -> None:
        ds._load_targeting()
        assert isinstance(ds.targeting, PandasDataFrame)
        assert isinstance(ds.weighted_targeting, PandasDataFrame)
        assert isinstance(ds.unweighted_targeting, PandasDataFrame)
        assert 'random' in ds.targeting.columns
        assert (ds.unweighted_targeting['weight'].values == np.ones(1000)).all()
        new_len = ds.weighted_targeting.drop_duplicates()['weight'].sum()
        assert ds.weighted_targeting.shape[0] == new_len

    @pytest.mark.unit_test
    def test_load_fairness(self, ds: Type[DataStore]) -> None:
        ds._load_fairness()
        assert isinstance(ds.fairness, PandasDataFrame)
        assert isinstance(ds.weighted_fairness, PandasDataFrame)
        assert isinstance(ds.unweighted_fairness, PandasDataFrame)
        assert 'random' in ds.fairness.columns
        assert (ds.unweighted_fairness['weight'].values == np.ones(1000)).all()
        new_len = ds.weighted_fairness.drop_duplicates()['weight'].sum()
        assert ds.weighted_fairness.shape[0] == new_len

    @pytest.mark.unit_test
    def test_load_survey(self, ds: Type[DataStore]) -> None:
        ds._load_survey()
        assert isinstance(ds.survey_data, PandasDataFrame)
        assert ds.survey_data.shape[0] == 1e3
        assert 'weight' in ds.survey_data.columns
        assert len(ds.survey_data.columns) == 34

        test_df = pd.DataFrame(data={'unique_id': ['XYZ'], 'bin0': [0], 'con0': [25]})
        ds._load_survey(dataframe=test_df)
        assert isinstance(ds.survey_data, PandasDataFrame)
        assert ds.survey_data.shape[0] == 1
        assert 'weight' in ds.survey_data.columns
        assert len(ds.survey_data.columns) == 4

    @pytest.mark.unit_test
    def test_merge(self, ds: Type[DataStore]) -> None:
        ds._load_features()
        ds._load_labels()
        ds.merge()

        assert isinstance(ds.merged, PandasDataFrame)
        assert ds.merged.shape[0] == 50

        assert isinstance(ds.x, PandasDataFrame)
        assert all(col not in ds.x.columns for col in ['name', 'label', 'weight'])
        assert len(ds.x.columns) == len(ds.merged.columns) - 3

        assert isinstance(ds.y, Series)
        assert isinstance(ds.weights, Series)
        assert ds.weights.min() >= 1

    @pytest.mark.unit_test
    @pytest.mark.parametrize("function, expected_error", [("_load_labels", ValueError),
                                                          ("_load_features", ValueError)])
    def test_merge_raises(self, ds: Type[DataStore], function, expected_error) -> None:
        with pytest.raises(expected_error):
            getattr(ds, function)()
            ds.merge()

    @pytest.mark.unit_test
    @pytest.mark.parametrize("data_type_map", [({DataType.CDR: None,
                                                 DataType.RECHARGES: None,
                                                 DataType.MOBILEDATA: None,
                                                 DataType.MOBILEMONEY: None,
                                                 DataType.ANTENNAS: None,
                                                 DataType.SHAPEFILES: None}),
                                               ({DataType.FEATURES: None,
                                                 DataType.LABELS: None}),
                                               ({DataType.CDR: None,
                                                 DataType.ANTENNAS: None,
                                                 DataType.SHAPEFILES: None,
                                                 DataType.HOME_GROUND_TRUTH: None,
                                                 DataType.POVERTY_SCORES: None})])
    def test_load_data(self, ds: Type[DataStore], data_type_map) -> None:
        ds.load_data(data_type_map)

    @pytest.mark.unit_test
    def test_filter_dates(self, ds: Type[DataStore]):
        # Load two datasets
        ds._load_recharges()
        ds._load_mobiledata()
        min_date, max_date = datetime(2020, 1, 1), datetime(2020, 2, 29)

        # Check that filtering with larger boundaries doesn't change anything
        ds.filter_dates(min_date - timedelta(days=1), max_date + timedelta(days=1))
        assert ds.recharges.agg(F.min('day')).collect()[0][0] == min_date
        assert ds.recharges.agg(F.max('day')).collect()[0][0] == max_date
        assert ds.mobiledata.agg(F.min('day')).collect()[0][0] == min_date
        assert ds.mobiledata.agg(F.max('day')).collect()[0][0] == max_date

        # Check that filtering with smaller boundaries works
        new_min_date, new_max_date = min_date + timedelta(days=1), max_date - timedelta(days=1)
        ds.filter_dates(new_min_date, new_max_date)
        assert ds.recharges.agg(F.min('day')).collect()[0][0] == new_min_date
        assert ds.recharges.agg(F.max('day')).collect()[0][0] == new_max_date
        assert ds.mobiledata.agg(F.min('day')).collect()[0][0] == new_min_date
        assert ds.mobiledata.agg(F.max('day')).collect()[0][0] == new_max_date

    @pytest.mark.unit_test
    @pytest.mark.parametrize("df, n_rows", [(pd.DataFrame(data={'caller_id': ['A', 'A'],
                                                                'volume': [50, 50],
                                                                'timestamp': ['2020-01-01 12:00:00',
                                                                              '2020-01-02 12:00:01']}),
                                             2),
                                            (pd.DataFrame(data={'caller_id': ['A', 'A'],
                                                                'volume': [50, 50],
                                                                'timestamp': ['2020-01-02 12:00:00',
                                                                              '2020-01-02 12:00:00']}),
                                             1)])
    def test_deduplicate(self, mock_dataframe_reader, ds: DataStore, df, n_rows):
        mock_dataframe_reader.return_value.csv.return_value = ds.spark.createDataFrame(df)
        ds._load_mobiledata()
        ds.deduplicate()
        assert ds.mobiledata.count() == n_rows

    @pytest.mark.unit_test
    @pytest.mark.parametrize("df, threshold, n_spammers", [(pd.DataFrame(data={'txn_type': ['call', 'call'],
                                                                               'caller_id': ['A', 'A'],
                                                                               'recipient_id': ['X', 'X'],
                                                                               'duration': [50, 50],
                                                                               'timestamp': ['2020-01-01 12:00:00',
                                                                                             '2020-01-02 12:00:01'],
                                                                               'international': ['domestic',
                                                                                                 'domestic']}),
                                                            1, 0),
                                                           (pd.DataFrame(data={'txn_type': ['call'] * 12,
                                                                               'caller_id': ['A', 'A'] + ['B'] * 10,
                                                                               'recipient_id': ['X'] * 12,
                                                                               'duration': [50, 50] + [50] * 10,
                                                                               'timestamp': [
                                                                                                '2020-01-02 12:00:00'] * 12,
                                                                               'international': ['domestic'] * 12}),
                                                            5, 1),
                                                           (pd.DataFrame(data={'txn_type': ['call'] * 12,
                                                                               'caller_id': ['A', 'A'] + ['B'] * 10,
                                                                               'recipient_id': ['X'] * 12,
                                                                               'duration': [50, 50] + [50] * 10,
                                                                               'timestamp': [
                                                                                                '2020-01-02 12:00:00'] * 12,
                                                                               'international': ['domestic'] * 12}),
                                                            10, 0)
                                                           ])
    def test_remove_spammers(self, mock_dataframe_reader, ds: DataStore, df, threshold, n_spammers):
        mock_dataframe_reader.return_value.csv.return_value = ds.spark.createDataFrame(df)
        ds._load_cdr()
        spammers = ds.remove_spammers(spammer_threshold=threshold)
        assert len(spammers) == n_spammers
        assert ds.cdr.where(col('caller_id').isin(spammers)).count() == 0

    @pytest.mark.unit_test
    def test_remove_spammers_raises(self, ds: DataStore):
        ds._load_recharges()
        with pytest.raises(ValueError):
            _ = ds.remove_spammers(spammer_threshold=1)

    timeseries_df = pd.DataFrame(data={'day': pd.date_range(start='2020-01-01', periods=6),
                                       'count': [10, 8, 9, 12, 8, 13]})

    @pytest.mark.unit_test
    @pytest.mark.parametrize("df, num_sds, n_outliers", [(timeseries_df, 2, 0),
                                                         (timeseries_df, 1.4, 1),
                                                         (timeseries_df, 0.9, 4)])
    def test_filter_outlier_days(self, mocker: MockerFixture, ds: DataStore, df, num_sds, n_outliers):
        mock_dataframe_reader = mocker.patch("cider.datastore.pd.read_csv", autospec=True)
        mock_dataframe_reader.return_value = df
        ds._load_cdr()
        outliers = ds.filter_outlier_days(num_sds=num_sds)
        assert len(outliers) == n_outliers

    survey_df = pd.DataFrame(data={'unique_id': [str(x) for x in range(10)],
                                   'con1': [1.7, 1.8, 0.9, 1.1, 0.5, 1.5, 1.0, 1.8, None, 0.1],
                                   'con2': [0.3, 0.9, 1.6, 0.9, 1.5, 1.1, 2.8, 0.8, 0.7, 1.4],
                                   'bin1': [0, 0, 1, 0, 0, 1, 1, 0, 1, 0]})

    @pytest.mark.unit_test
    @pytest.mark.parametrize("df, cols, num_sds, outliers", [(survey_df, ['con1'], 1, ['1', '4', '7', '9']),
                                                             (survey_df, ['con1', 'con2'], 1,
                                                              ['0', '1', '4', '6', '7', '9']),
                                                             (survey_df, ['con1', 'con2'], 2.5, [])])
    def test_remove_survey_outliers(self, mocker: MockerFixture, ds: DataStore, df, cols, num_sds, outliers):
        mock_dataframe_reader = mocker.patch("cider.datastore.pd.read_csv", autospec=True)
        mock_dataframe_reader.return_value = df
        ds._load_survey()
        assert set(outliers) == ds.remove_survey_outliers(cols=cols, num_sds=num_sds)

    @pytest.mark.unit_test
    @pytest.mark.parametrize("df, cols, expected_exception", [(survey_df, ['bin1'], TypeError),
                                                              (survey_df, ['con3'], ValueError)])
    def test_remove_survey_outliers_raises(self, mocker: MockerFixture, ds: DataStore, df, cols, expected_exception):
        mock_dataframe_reader = mocker.patch("cider.datastore.pd.read_csv", autospec=True)
        mock_dataframe_reader.return_value = df
        ds._load_survey()
        with pytest.raises(expected_exception):
            ds.remove_survey_outliers(cols=cols)

    @pytest.mark.unit_test
    def test_remove_survey_outliers_raises_before_load(self, ds: DataStore, ):
        with pytest.raises(ValueError):
            ds.remove_survey_outliers(cols=['con1'])

    # TODO: Write integration tests
    @pytest.mark.integration_test
    @pytest.mark.skip(reason="Test not yet implemented")
    def test_datastore_end_to_end(self, datastore_class: Type[DataStore], ds_mock_spark: DataStore) -> None:
        pass


class TestOptDatastoreClass:
    """All the tests related to object that implement OptDatastore."""

    @pytest.fixture()
    def ds(self) -> OptDataStore:
        out = OptDataStore(cfg_dir="configs/test_config.yml")
        return out

    @pytest.mark.unit_test
    @pytest.mark.parametrize("data_type_map, n_users", [({DataType.CDR: None}, 1000),
                                                        ({DataType.MOBILEMONEY: None}, 700),
                                                        ({DataType.CDR: None,
                                                          DataType.MOBILEMONEY: None}, 1000)])
    def test_initialize_user_consent_table(self, ds: OptDataStore, data_type_map, n_users):
        assert getattr(ds, 'user_consent', None) is None
        ds.load_data(data_type_map=data_type_map)
        ds.initialize_user_consent_table()
        assert getattr(ds, 'user_consent', None) is not None
        assert ds.user_consent.count() == n_users

    @pytest.mark.unit_test
    @pytest.mark.parametrize("data", [({'user_id': ['WzwHpoldPp', 'xkThzuCDAY', 'OtFfOxcGMu', 'pjjlDwunYH']}),
                                      ({'user_id': ['WzwHpoldPp', 'xkThzuCDAY', 'OtFfOxcGMu', 'pjjlDwunYH'],
                                        'include': [True, False, False, True]})])
    def test_initialize_user_consent_table_from_file(self, mocker: MockerFixture, ds: OptDataStore, data):
        mock_dataframe_reader = mocker.patch("cider.datastore.pd.read_csv", autospec=True)
        df = pd.DataFrame(data=data)
        mock_dataframe_reader.return_value = df
        ds.load_data(data_type_map={DataType.RECHARGES: None,
                                    DataType.MOBILEMONEY: None})
        ds.initialize_user_consent_table(read_from_file=True)
        if len(data) == 1:
            assert ds.recharges.where(col('caller_id').isin(data['user_id'])).count() == ds.recharges.count()
            assert ds.mobilemoney.where(col('caller_id').isin(data['user_id'])).count() == ds.mobilemoney.count()
        else:
            included = list(df[df['include'] == True]['user_id'].values)
            excluded = list(df[df['include'] == False]['user_id'].values)
            assert ds.recharges.where(col('caller_id').isin(excluded)).count() == 0
            assert ds.mobilemoney.where(col('caller_id').isin(excluded)).count() == 0
            assert ds.recharges.where(col('caller_id').isin(included)).count() == ds.recharges.count()
            assert ds.mobilemoney.where(col('caller_id').isin(included)).count() == ds.mobilemoney.count()

    @pytest.mark.unit_test
    @pytest.mark.parametrize("data, expected_exception",
                             [({'user_idx': ['WzwHpoldPp', 'xkThzuCDAY', 'OtFfOxcGMu', 'pjjlDwunYH']}, ValueError),
                              ({'user_id': ['WzwHpoldPp', 'xkThzuCDAY', 'OtFfOxcGMu', 'pjjlDwunYH'],
                                'opted_in': [True, False, False, True]}, ValueError),
                              ({'user_id': ['WzwHpoldPp', 'xkThzuCDAY', 'OtFfOxcGMu', 'pjjlDwunYH'],
                                'include': ['yes', 'no', 'no', 'yes']}, ValueError)
                              ])
    def test_initialize_user_consent_table_from_file_raises(self, mocker: MockerFixture, ds: OptDataStore, data,
                                                            expected_exception):
        mock_dataframe_reader = mocker.patch("cider.datastore.pd.read_csv", autospec=True)
        df = pd.DataFrame(data=data)
        mock_dataframe_reader.return_value = df
        ds.load_data(data_type_map={DataType.RECHARGES: None,
                                    DataType.MOBILEMONEY: None})
        with pytest.raises(expected_exception):
            ds.initialize_user_consent_table(read_from_file=True)

    # TODO: write opt_in/out tests
