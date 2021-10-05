from cider.datastore import DataStore, InitializerInterface, OptDataStore
from helpers.utils import get_project_root
from pandas import DataFrame as PandasDataFrame
import pytest
from unittest.mock import patch
import os

@pytest.fixture()
def dummy_data_frame() -> PandasDataFrame:
    """Fixture that is populated with some test data."""
    return PandasDataFrame()


@pytest.mark.parametrize("datastore_class", [DataStore, OptDataStore])
class TestInitializerInterfaceClasses():
    """All the tests related to datastore objects for which the DataStore and OptDataStore should have the same behavior."""

    @pytest.mark.parametrize("config_file_path", ["configs/config_emily.yml", "configs/config_lucio.yml", "configs/config_min.yml", "configs/config.yml"])
    def test_config_datastore(self, config_file_path: str, datastore_class: InitializerInterface):
        """Test that each config file is not stale and can initialize without raising an error."""
        datastore = datastore_class(cfg_dir=os.path.join(get_project_root(), config_file_path))
    
    @pytest.mark.parametrize("config_file_path,expected_exception", [("", FileNotFoundError), ("\\malformed#$directory!!!(38", FileNotFoundError)])
    def test_config_datastore_exception(self, config_file_path: str, datastore_class: InitializerInterface, expected_exception: Exception):
        with pytest.raises(expected_exception):
            datastore = datastore_class(cfg_dir=config_file_path)
    
    @patch('pyspark.sql.SparkSession')
    @pytest.fixture()
    def ds(self, datastore_class: InitializerInterface, mock_spark):
        """A fully configured data store to be used in tests."""
        # TODO: Perhaps decouple the creation of this object from config files altogether or make a test_config.yml
        # I would lobby for having an intermediate dataclass that represents the config file as a python object with known semantics


        # Also here is an opportunity to give an example of mocking an object that your unit test would use
        # See https://docs.python.org/3/library/unittest.mock.html
        # See https://myadventuresincoding.wordpress.com/2011/02/26/python-python-mock-cheat-sheet/
        mock_spark = mock_spark.return_value.read.csv.return_value = {"col1": (0,1,2,3)}
        # Now this object will have a mock spark, since we are trying to unit test our code, not test spark
        out = datastore_class(cfg_dir="configs/config.yml")
        # Can test for example that the mock was used
        assert mock_spark.called()
        return out

        
    # TODO: Same test for antennas, recharges, mobiledata, mobilemoney, shapefiles, home_group_truth, poverty_scores, features, labels, targeting, fairness, wealth map
    # merge, load_data, filter_dates, deduplicate, remove_spammers, filter_outlier_days
    @pytest.mark.skip(reason="Test not yet implemented")
    def test_load_cdr(self, ds: InitializerInterface):
        # TODO: Add asserts for the following:
        # TODO: Test successful operation: nominal case, edge cases, test None when anything is Optional, test for idempotency where appropriate, test zero length iterables
        # TODO: Test expected failures raise appropriate errors: Malformed inputs, invalid inputs, basically any code path that should raise an exception
        pass

    # Example where classes have the same expected outputs
    test_example_same_behavior_per_class_data = [
        (0, 1, 0),
        (2, 4, 8),
    ]
    @pytest.mark.parametrize("a, b, expected", test_example_same_behavior_per_class_data)
    def test_example_same_behavior_per_class(self, datastore_class, a, b, expected):

        def prentend_this_function_is_a_class_function(a,b):
            out = a*b
            return out
        
        assert prentend_this_function_is_a_class_function(a,b) == expected

    # Example where classes have different expected outputs
    test_example_different_behavior_per_class_data = [
        (0, 1, {DataStore: 0, OptDataStore: 1}),
        (2, 4, {DataStore: 8, OptDataStore: 6}),
    ]
    @pytest.mark.parametrize("a, b, expected", test_example_different_behavior_per_class_data)
    def test_example_different_behavior_per_class(self, datastore_class, a, b, expected):

        def prentend_this_function_is_a_class_function(a,b):
            if datastore_class == DataStore:
                out = a*b
            else:
                out = a+b
            return out
        
        assert prentend_this_function_is_a_class_function(a,b) == expected[datastore_class]

