import os

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from typing import MutableMapping, Type

import pytest
from pytest_mock import MockerFixture

from cider.datastore import DataStore, OptDataStore
from helpers.utils import get_project_root


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
    def ds(self, mocker: MockerFixture, datastore_class: Type[DataStore]) -> DataStore:
        # TODO: Perhaps decouple the creation of this object from config files altogether or make a test_config.yml
        # I would lobby for having an intermediate dataclass that represents the config file as a python object with known semantics
        out = datastore_class(cfg_dir="configs/config.yml")
        return out

    # TODO: Same test for antennas, recharges, mobiledata, mobilemoney, shapefiles, home_group_truth, poverty_scores, features, labels, targeting, fairness, wealth map
    # merge, load_data, filter_dates, deduplicate, remove_spammers, filter_outlier_days
    @pytest.mark.unit_test
    def test_load_cdr(self, ds: Type[DataStore]) -> None:  # ds_mock_spark: DataStore
        # TODO: Add asserts for the following:
        # TODO: Test successful operation: nominal case, edge cases, test None when anything is Optional, test for idempotency where appropriate, test zero length iterables
        # TODO: Test expected failures raise appropriate errors: Malformed inputs, invalid inputs, basically any code path that should raise an exception
        ds._load_cdr()
        assert type(ds.cdr) == SparkDataFrame
        assert ds.cdr.count() == 1e5
        assert 'caller_id' in ds.cdr.columns

        # check incorrect input df
        with pytest.raises(TypeError):
            ds._load_cdr(dataframe=5)

        # check missing columns
        test_df = pd.DataFrame(data={'txn_type': ['text'], 'caller_id': ['A'], 'recipient_id': ['B'],
                                     'timestamp': ['2021-01-01']})
        with pytest.raises(ValueError):
            ds._load_cdr(dataframe=test_df)

        # check wrong column value
        test_df = pd.DataFrame(data={'txn_type': ['text_message'], 'caller_id': ['A'], 'recipient_id': ['B'],
                                     'timestamp': ['2021-01-01'], 'duration': [60], 'international': ['domestic']})
        with pytest.raises(ValueError):
            ds._load_cdr(dataframe=test_df)

        test_df = pd.DataFrame(data={'txn_type': ['text'], 'caller_id': ['A'], 'recipient_id': ['B'],
                                     'timestamp': ['2021-01-01'], 'duration': [60], 'international': ['domestic']})
        ds._load_cdr(dataframe=test_df)
    
    
    malformed_cdr_dataframes_and_errors = [
        (pd.DataFrame(data={'txn_type': ['text'], 'caller_id': ['A'], 'recipient_id': ['B'], 'timestamp': ['2021-01-01']}), ValueError),
    ]
    
    @pytest.mark.unit_test
    @pytest.mark.parametrize("dataframe, expected_error", malformed_cdr_dataframes_and_errors)
    def test_load_cdr_raises_from_csv(self, mocker: MockerFixture, ds: DataStore, dataframe, expected_error):
        mock_spark = mocker.patch("helpers.utils.SparkSession", autospec=True)
        mock_read_csv = mock_spark.return_value.read.csv
        mock_read_csv.return_value = dataframe   
        with pytest.raises(expected_error):
            ds._load_cdr()

    @pytest.mark.parametrize("dataframe, expected_error", malformed_cdr_dataframes_and_errors)
    def test_load_cdr_raises_from_csv(self, ds: DataStore, dataframe, expected_error):
        with pytest.raises(expected_error):
            ds._load_cdr(dataframe=dataframe)


    @pytest.mark.unit_test
    def test_load_antennas(self, ds: Type[DataStore]) -> None:  # ds_mock_spark: DataStore
        # TODO: Add asserts for the following:
        # TODO: Test successful operation: nominal case, edge cases, test None when anything is Optional, test for idempotency where appropriate, test zero length iterables
        # TODO: Test expected failures raise appropriate errors: Malformed inputs, invalid inputs, basically any code path that should raise an exception
        ds._load_antennas()
        assert type(ds.antennas) == SparkDataFrame
        assert ds.antennas.count() == 297
        assert 'antenna_id' in ds.antennas.columns

        # check incorrect input df
        with pytest.raises(TypeError):
            ds._load_antennas(dataframe=5)

        # check missing columns
        test_df = pd.DataFrame(data={'antenna_id': ['1'], 'latitude': ['10']})
        with pytest.raises(ValueError):
            ds._load_antennas(dataframe=test_df)

        test_df = pd.DataFrame(data={'antenna_id': ['1'], 'latitude': ['10'], 'longitude': ['25.3']})
        ds._load_antennas(dataframe=test_df)

    @pytest.mark.unit_test
    def test_load_recharges(self, ds: Type[DataStore]) -> None:  # ds_mock_spark: DataStore
        # TODO: Test successful operation: nominal case, edge cases, test None when anything is Optional, test for idempotency where appropriate, test zero length iterables
        # TODO: Test expected failures raise appropriate errors: Malformed inputs, invalid inputs, basically any code path that should raise an exception
        ds._load_recharges()
        assert type(ds.recharges) == SparkDataFrame
        assert ds.recharges.count() == 1e4
        assert 'amount' in ds.recharges.columns

        # check incorrect input df
        with pytest.raises(TypeError):
            ds._load_recharges(dataframe=5)

        test_df = pd.DataFrame(data={'caller_id': ['A'], 'amount': ['100'], 'timestamp': ['2020-01-01']})
        ds._load_recharges(dataframe=test_df)


    @pytest.mark.integration_test
    @pytest.mark.skip(reason="Test not yet implemented")
    def test_datastore_end_to_end(self, datastore_class: Type[DataStore], ds_mock_spark: DataStore) -> None:
        pass

    # Example where classes have the same expected outputs
    test_example_same_behavior_per_class_data = [
        (0, 1, 0),
        (2, 4, 8),
    ]

    @pytest.mark.unit_test
    @pytest.mark.parametrize(
        "a, b, expected", test_example_same_behavior_per_class_data
    )
    def test_example_same_behavior_per_class(
        self, datastore_class: Type[DataStore], a: int, b: int, expected: int
    ) -> None:
        def prentend_this_function_is_a_class_function(a: int, b: int) -> int:
            out = a * b
            return out

        assert prentend_this_function_is_a_class_function(a, b) == expected

    # Example where classes have different expected outputs
    test_example_different_behavior_per_class_data = [
        (0, 1, {DataStore: 0, OptDataStore: 1}),
        (2, 4, {DataStore: 8, OptDataStore: 6}),
    ]

    @pytest.mark.unit_test
    @pytest.mark.parametrize(
        "a, b, expected", test_example_different_behavior_per_class_data
    )
    def test_example_different_behavior_per_class(
        self,
        datastore_class: Type[DataStore],
        a: int,
        b: int,
        expected: MutableMapping[Type[DataStore], int],
    ) -> None:
        def prentend_this_function_is_a_class_function(a: int, b: int) -> int:
            if datastore_class == DataStore:
                out = a * b
            else:
                out = a + b
            return out

        assert (
            prentend_this_function_is_a_class_function(a, b)
            == expected[datastore_class]
        )
