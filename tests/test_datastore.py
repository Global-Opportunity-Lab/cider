import os

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.utils import AnalysisException
from typing import MutableMapping, Type

import pytest
from pytest_mock import MockerFixture

from cider.datastore import DataStore, OptDataStore
from helpers.utils import get_project_root, get_spark_session

malformed_dataframes_and_errors = {
    'cdr': [(pd.DataFrame(
                data={'txn_type': ['text'], 'caller_id': ['A'], 'recipient_id': ['B'], 'timestamp': ['2021-01-01']}),
             ValueError),
            (pd.DataFrame(data={'txn_type': ['text_message'], 'caller_id': ['A'], 'recipient_id': ['B'],
                                'timestamp': ['2021-01-01'], 'duration': [60], 'international': ['domestic']}),
             ValueError)],
    'antennas': [(pd.DataFrame(data={'antenna_id': ['1'], 'latitude': ['10']}), ValueError)],
    'recharges': [(pd.DataFrame(data={'caller_id': ['A'], 'amount': ['2021-01-01']}), AnalysisException)],
    'mobiledata': [(pd.DataFrame(data={'caller_id': ['A'], 'timestamp': ['2021-01-01']}), AnalysisException)]
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
    def ds_mock_spark(self, mocker: MockerFixture, datastore_class: Type[DataStore]) -> DataStore:
        # TODO: Perhaps decouple the creation of this object from config files altogether or make a test_config.yml
        # I would lobby for having an intermediate dataclass that represents the config file as a python object with known semantics

        # Also here is an opportunity to give an example of mocking an object that your unit test would use
        mock_spark = mocker.patch("helpers.utils.SparkSession", autospec=True)
        mock_read_csv = mock_spark.return_value.read.csv
        mock_read_csv.return_value = {"col1": (0, 1, 2, 3)}
        # Now this object will have a mock spark, since we are trying to unit test our code, not test spark
        out = datastore_class(cfg_dir="configs/config.yml")
        # Can test for example that the mock was used
        assert mock_read_csv.called
        return out


    @pytest.fixture()
    def ds(self, datastore_class: Type[DataStore]) -> DataStore:
        # I would lobby for having an intermediate dataclass that represents the config file as a python object with known semantics
        out = datastore_class(cfg_dir="configs/config.yml")
        return out

    @pytest.fixture()
    def spark(self, ds: Type[DataStore]) -> DataStore:
        # I would lobby for having an intermediate dataclass that represents the config file as a python object with known semantics
        spark = get_spark_session(ds.cfg)
        yield spark

    @pytest.mark.unit_test
    def test_load_cdr(self, ds: Type[DataStore]) -> None:  # ds_mock_spark: DataStore
        # TODO: Add asserts for the following:
        # TODO: Test successful operation: nominal case, edge cases, test None when anything is Optional, test for idempotency where appropriate, test zero length iterables
        # TODO: Test expected failures raise appropriate errors: Malformed inputs, invalid inputs, basically any code path that should raise an exception
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
    def test_load_cdr_raises_from_csv(self, mocker: MockerFixture, ds: DataStore, spark, dataframe, expected_error):
        mock_spark = mocker.patch("helpers.utils.SparkSession", autospec=True)
        mock_read_csv = mock_spark.return_value.read.csv
        mock_read_csv.return_value = spark.createDataFrame(dataframe)
        with pytest.raises(expected_error):
            ds._load_cdr()

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['cdr'])
    def test_load_cdr_raises_from_df(self, ds: DataStore, dataframe, expected_error):
        with pytest.raises(expected_error):
            ds._load_cdr(dataframe=dataframe)

    @pytest.mark.unit_test
    def test_load_antennas(self, ds: Type[DataStore]) -> None:
        # TODO: Test successful operation: nominal case, edge cases, test None when anything is Optional, test for idempotency where appropriate, test zero length iterables
        # TODO: Test expected failures raise appropriate errors: Malformed inputs, invalid inputs, basically any code path that should raise an exception
        ds._load_antennas()
        assert type(ds.antennas) == SparkDataFrame
        assert ds.antennas.count() == 297
        assert dict(ds.antennas.dtypes)['latitude'] == 'float'
        assert len(ds.antennas.columns) == 4

        test_df = pd.DataFrame(data={'antenna_id': ['1'], 'latitude': ['10'], 'longitude': ['25.3']})
        ds._load_antennas(dataframe=test_df)
        assert type(ds.antennas) == SparkDataFrame
        assert ds.antennas.count() == 1
        assert dict(ds.antennas.dtypes)['latitude'] == 'float'
        assert len(ds.antennas.columns) == 3

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['antennas'])
    def test_load_antennas_raises_from_csv(self, mocker: MockerFixture, ds: DataStore, spark, dataframe, expected_error):
        mock_spark = mocker.patch("helpers.utils.SparkSession", autospec=True)
        mock_read_csv = mock_spark.return_value.read.csv
        mock_read_csv.return_value = spark.createDataFrame(dataframe)
        with pytest.raises(expected_error):
            ds._load_antennas()

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['antennas'])
    def test_load_antennas_raises_from_df(self, ds: DataStore, dataframe, expected_error):
        with pytest.raises(expected_error):
            ds._load_antennas(dataframe=dataframe)

    @pytest.mark.unit_test
    def test_load_recharges(self, ds: Type[DataStore]) -> None:  # ds_mock_spark: DataStore
        # TODO: Test successful operation: nominal case, edge cases, test None when anything is Optional, test for idempotency where appropriate, test zero length iterables
        # TODO: Test expected failures raise appropriate errors: Malformed inputs, invalid inputs, basically any code path that should raise an exception
        ds._load_recharges()
        assert type(ds.recharges) == SparkDataFrame
        assert ds.recharges.count() == 1e4
        assert len(ds.recharges.columns) == 4

        test_df = pd.DataFrame(data={'caller_id': ['A'], 'amount': ['100'], 'timestamp': ['2020-01-01']})
        ds._load_recharges(dataframe=test_df)
        assert type(ds.recharges) == SparkDataFrame
        assert ds.recharges.count() == 1
        assert len(ds.recharges.columns) == 4

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['recharges'])
    def test_load_recharges_raises_from_csv(self, mocker: MockerFixture, ds: DataStore, spark, dataframe, expected_error):
        mock_spark = mocker.patch("helpers.utils.SparkSession", autospec=True)
        mock_read_csv = mock_spark.return_value.read.csv
        mock_read_csv.return_value = dataframe
        with pytest.raises(expected_error):
            ds._load_recharges()

    @pytest.mark.unit_test
    def test_load_recharges_using_mocker(self, mocker: MockerFixture, ds: DataStore, spark):
        mock_spark = mocker.patch("helpers.utils.SparkSession", autospec=True)
        mock_read_csv = mock_spark.return_value.read.csv
        dataframe = pd.DataFrame(data={'caller_id': ['A'], 'amount': [100], 'timestamp': ['2021-01-01']})
        mock_read_csv.return_value = spark.createDataFrame(dataframe)
        ds._load_recharges()
        assert ds.recharges == spark.createDataFrame(dataframe)

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['recharges'])
    def test_load_recharges_raises_from_df(self, ds: DataStore, dataframe, expected_error):
        with pytest.raises(expected_error):
            ds._load_recharges(dataframe=dataframe)

    @pytest.mark.unit_test
    def test_load_mobiledata(self, ds: Type[DataStore]) -> None:  # ds_mock_spark: DataStore
        # TODO: Test successful operation: nominal case, edge cases, test None when anything is Optional, test for idempotency where appropriate, test zero length iterables
        # TODO: Test expected failures raise appropriate errors: Malformed inputs, invalid inputs, basically any code path that should raise an exception
        ds._load_mobiledata()
        assert type(ds.mobiledata) == SparkDataFrame
        assert ds.mobiledata.count() == 1e4
        assert len(ds.mobiledata.columns) == 4

        test_df = pd.DataFrame(data={'caller_id': ['A'], 'volume': ['100'], 'timestamp': ['2020-01-01']})
        ds._load_mobiledata(dataframe=test_df)
        assert type(ds.mobiledata) == SparkDataFrame
        assert ds.mobiledata.count() == 1
        assert len(ds.mobiledata.columns) == 4

    @pytest.mark.unit_test
    @pytest.mark.skip(reason="Test not yet implemented")
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['mobiledata'])
    def test_load_mobiledata_raises_from_csv(self, mocker: MockerFixture, ds: DataStore, spark, dataframe, expected_error):
        mock_spark = mocker.patch("helpers.utils.SparkSession", autospec=True)
        mock_read_csv = mock_spark.return_value.read.csv
        mock_read_csv.return_value = spark.createDataFrame(dataframe)
        with pytest.raises(expected_error):
            ds._load_mobiledata()

    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_dataframes_and_errors['mobiledata'])
    def test_load_mobiledata_raises_from_df(self, ds: DataStore, dataframe, expected_error):
        with pytest.raises(expected_error):
            ds._load_mobiledata(dataframe=dataframe)


    @pytest.mark.integration_test
    @pytest.mark.skip(reason="Test not yet implemented")
    def test_datastore_end_to_end(self, datastore_class: Type[DataStore], ds_mock_spark: DataStore) -> None:
        pass
