import pytest

from app_config import ConfigError, validate_config


def test_existing_config_gets_safe_archive_defaults():
    cfg = validate_config({})['ocr']
    assert cfg['backend'] == 'paddle'
    assert cfg['cpu_threads'] == 0
    assert cfg['onnx_cpu_mem_arena'] is False
    assert cfg['onnx_mem_pattern'] is False
    assert cfg['cache_enabled'] is True


def test_onnx_requires_validated_cpu_medium_profile():
    valid = {'ocr': {'backend': 'onnxruntime', 'device': 'cpu', 'model_profile': 'medium'}}
    assert validate_config(valid)['ocr']['backend'] == 'onnxruntime'
    for patch in [{'device': 'gpu:0'}, {'model_profile': 'tiny'}, {'version': 'PP-OCRv5'}]:
        with pytest.raises(ConfigError):
            validate_config({'ocr': {**valid['ocr'], **patch}})


@pytest.mark.parametrize('patch', [
    {'backend': 'unknown'}, {'cpu_threads': True}, {'cpu_threads': -1}, {'cpu_threads': 129},
    {'onnx_cpu_mem_arena': 'false'}, {'cache_enabled': 1}, {'cache_max_mb': 0},
    {'cache_ttl_days': 0}, {'det_limit_side_len': 100},
])
def test_archive_options_reject_invalid_values(patch):
    with pytest.raises(ConfigError):
        validate_config({'ocr': patch})
