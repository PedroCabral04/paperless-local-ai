import pytest
from ocr_runtime import backend_options, runtime_signature


def test_onnx_options_are_explicit_and_cpu_only():
    cfg={'backend':'onnxruntime','version':'PP-OCRv6','model_profile':'medium','device':'cpu'}
    options=backend_options(cfg,4,model_dirs={'text_detection_model_dir':'synthetic-det','text_recognition_model_dir':'synthetic-rec'})
    assert options['engine']=='onnxruntime'
    assert options['enable_hpi'] is False
    assert options['engine_config']['providers']==['CPUExecutionProvider']
    assert options['engine_config']['enable_mem_pattern'] is False
    assert options['engine_config']['intra_op_num_threads']==4
    with pytest.raises(ValueError):backend_options({**cfg,'device':'gpu:0'},4,model_dirs={})


def test_paddle_settings_remain_upstream_owned():
    assert backend_options({'backend':'paddle'},4)=={}


def test_cache_signature_changes_with_recognition_settings_not_retention():
    a=runtime_signature({'backend':'onnxruntime','cpu_threads':2,'cache_ttl_days':30})
    assert a==runtime_signature({'backend':'onnxruntime','cpu_threads':2,'cache_ttl_days':60})
    assert a!=runtime_signature({'backend':'onnxruntime','cpu_threads':4,'cache_ttl_days':30})


def test_native_model_artifact_change_invalidates_cache_signature(tmp_path, monkeypatch):
    monkeypatch.setenv('PADDLE_PDX_CACHE_HOME', str(tmp_path))
    model = tmp_path/'official_models'/'PP-OCRv6_medium_rec'
    model.mkdir(parents=True)
    weights = model/'inference.pdiparams'
    weights.write_bytes(b'synthetic weights one')
    config = {'backend':'paddle','version':'PP-OCRv6','model_profile':'medium'}
    before = runtime_signature(config)
    weights.write_bytes(b'synthetic weights changed')
    assert runtime_signature(config) != before


def test_comparison_signature_reports_effective_hpi_setting(monkeypatch):
    from compare_ocr import comparison_signature
    monkeypatch.setenv('OCR_ENABLE_HPI', 'true')
    assert comparison_signature({'backend':'paddle'})['enable_hpi'] == 'false'
