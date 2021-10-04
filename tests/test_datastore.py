from cider.datastore import DataStore, OptDataStore
from helpers.utils import get_project_root
from pandas import DataFrame as PandasDataFrame
import pytest
import os

@pytest.fixture()
def dummy_data_frame() -> PandasDataFrame:
    """Fixture that is populated with some test data."""
    # TODO: Acutally add data
    return PandasDataFrame()


@pytest.mark.parametrize("config_file_path", ["configs/config_emily.yml", "configs/config_lucio.yml", "configs/config_min.yml", "configs/config.yml"])
@pytest.mark.parametrize("datastore_class", [DataStore, OptDataStore])
def test_config_datastore(config_file_path: str, datastore_class: DataStore):
    """Test that each config file is not stale and can initialize without raising an error."""
    datastore = datastore_class(cfg_dir=os.path.join(get_project_root(), config_file_path))

#TODO: Tests for every member function including 1) A variety of inputs, especially None if the input is Optional. Tests that they raise an error when appropriate