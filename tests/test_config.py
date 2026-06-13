import pytest
from src.utils.config import CONFIG


def test_config_loads():
    """Тест проверяет, что конфиг успешно загружается"""
    config = CONFIG

    assert config is not None
    assert hasattr(config, 'stage1')
    assert hasattr(config, 'stage2')
    assert hasattr(config, 'stage3')
    assert hasattr(config, 'RANDOM_SEED')


def test_config_paths():
    """Проверяем, что пути настроены корректно"""
    paths = CONFIG.stage1.paths
    assert paths.DATA_DIR is not None
    assert paths.CHUNKS_DIR is not None
