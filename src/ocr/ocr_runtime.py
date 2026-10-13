"""Explicit inference backend options and immutable ONNX model provenance."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

MODELS = {
    'det': {
        'repo': 'PaddlePaddle/PP-OCRv6_medium_det_onnx',
        'revision': '61323801669c338b7891481ec7bac61ce31b576a',
        'files': {
            'inference.json': '0f1a7ec35da36173529c7a60238b7f7919e3831929c3f700ad90ad4896adecd5',
            'inference.yml': '7298d5ead546584af2504d03355f881ac7a7bc0eb1e282d3e159277c1d0af871',
            'inference.onnx': 'eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1',
        },
    },
    'rec': {
        'repo': 'PaddlePaddle/PP-OCRv6_medium_rec_onnx',
        'revision': '50c7eacafc52fa7bcf4194e8cd08e46f8558504b',
        'files': {
            'inference.json': '0b2e25e990bd072f1bf77d59d67d508bce6c4bd44af6624e0fb27d6da2cd00e8',
            'inference.yml': '991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129',
            'inference.onnx': '9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba',
        },
    },
}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_onnx_models(root: Path | None = None) -> dict[str, str]:
    import requests
    root = root or Path(os.getenv('OCR_ONNX_MODEL_DIR', '/ocr-data/onnx'))
    result = {}
    for part, model in MODELS.items():
        folder = root / model['repo'].split('/')[-1] / model['revision']
        folder.mkdir(parents=True, exist_ok=True)
        for name, expected in model['files'].items():
            dest = folder / name
            if dest.exists():
                if file_digest(dest) != expected:
                    raise ValueError('ONNX model checksum mismatch')
                continue
            descriptor, temporary = tempfile.mkstemp(prefix=name+'.', suffix='.part', dir=folder)
            try:
                url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{name}"
                with os.fdopen(descriptor, 'wb') as output, requests.get(url, stream=True, timeout=(15, 180)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if file_digest(Path(temporary)) != expected:
                    raise ValueError('Downloaded ONNX model checksum mismatch')
                os.replace(temporary, dest)
            finally:
                Path(temporary).unlink(missing_ok=True)
        key = 'text_detection_model_dir' if part == 'det' else 'text_recognition_model_dir'
        result[key] = str(folder)
    return result


def backend_options(config: dict, threads: int, *, model_dirs: dict | None = None) -> dict:
    """Only explicit ONNX opts override upstream Paddle/HPI behavior."""
    if config.get('backend', 'paddle') != 'onnxruntime':
        return {}
    if config.get('version') != 'PP-OCRv6' or config.get('model_profile') != 'medium' or config.get('device') != 'cpu':
        raise ValueError('ONNX currently requires PP-OCRv6 Medium on cpu')
    return {
        'engine': 'onnxruntime', 'enable_hpi': False, 'enable_mkldnn': False,
        'engine_config': {'intra_op_num_threads': threads, 'inter_op_num_threads': 1,
                          'providers': ['CPUExecutionProvider'],
                          'enable_cpu_mem_arena': config.get('onnx_cpu_mem_arena', False),
                          'enable_mem_pattern': config.get('onnx_mem_pattern', False)},
        **(model_dirs if model_dirs is not None else resolve_onnx_models()),
    }


@lru_cache(maxsize=32)
def _model_hash(path: str, size: int, modified_ns: int) -> str:
    return file_digest(Path(path))


def paddle_model_fingerprint(config: dict) -> dict:
    """Hashes of downloaded model artifacts, refreshed when their stat changes."""
    root = Path(os.getenv('PADDLE_PDX_CACHE_HOME', str(Path.home()/'.paddlex'))) / 'official_models'
    result = {}
    for folder in sorted(root.glob('*')):
        # Paddle v6 names are explicit; older upstream defaults use a broader inventory.
        if config.get('version') == 'PP-OCRv6' and not folder.name.startswith(f"PP-OCRv6_{config.get('model_profile', 'medium')}_"):
            continue
        for file in sorted(folder.glob('inference.*')):
            stat = file.stat()
            result[folder.name+'/'+file.name] = _model_hash(str(file),stat.st_size,stat.st_mtime_ns)
    return result


def runtime_signature(config: dict) -> dict:
    """Cache invalidates on config, installed runtime, models or source changes."""
    packages = {}
    for package in ['paddleocr', 'paddlex', 'paddlepaddle', 'onnxruntime', 'ultra-infer-python', 'numpy', 'opencv-contrib-python']:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = 'not-installed'
    sources = {name: file_digest(Path(__file__).with_name(name)) for name in ['ocr_runtime.py', 'service.py', 'ocr_page_cache.py']}
    return {'format': 1, 'runtime': packages, 'source': sources,
            'models': MODELS if config.get('backend') in ('onnxruntime','openvino') else paddle_model_fingerprint(config),
            'enable_hpi': os.getenv('OCR_ENABLE_HPI', 'false'),
            'thread_override': os.getenv('OCR_CPU_THREADS', ''),
            'ocr': {k: v for k, v in config.items() if not k.startswith('cache_')}}
