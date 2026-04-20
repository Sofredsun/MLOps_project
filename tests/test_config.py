from src.utils.config import CONFIG

def test_config_loads():
    config = CONFIG()
    assert config is not None