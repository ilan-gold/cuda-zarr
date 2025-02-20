import pytest
from zarr import config


@pytest.fixture(autouse=True)
def _set_config():
    config.enable_gpu()
