# Archive OCR fork — implementation plan

Preserve upstream's OCRmyPDF integration, two application images and shared AI lock. Paperless owns text search. No layout detector, new document manager or native-parser rewrite is required.

## Deliverables and acceptance

1. Fork with upstream remote and isolated branch; retain MIT notices and upstream capabilities.
2. Explicit Paddle / ONNX CPU backend and memory/thread controls in versioned configuration and Control Center. Backward-compatible defaults; unsupported combinations rejected before inference.
3. Persistent bounded page-result cache keyed by image content, effective OCR configuration, model/runtime/source fingerprints. Atomic commit, integrity checking, expiry, concurrent duplicate serialization; failures never cached. Restart reuses completed inference, not necessarily rasterization/PDF assembly.
4. Historical-scan deployment preset that replaces pre-existing OCR while preserving the uploaded original; real Paperless reprocess/search/hash test.
5. Authenticated admin proxy deployment with hidden direct Control Center port and no credentials in Git/browser config. Protect admin APIs as well as HTML; keep Paperless's existing authenticated relay working.
6. Advisory blank/repetition/low-confidence diagnostics; no claim that confidence proves accuracy. Persist bounded request statistics/history without OCR content in operational logs.
7. Control Center backend/cache/resource controls and request/cache/diagnostic monitoring. Do not pretend to know unreceived documents or remaining PDF pages.
8. Read-only OCR comparison command on selected input images, with explicit backend/profile overrides, timing/memory and private outputs. No writes to Paperless or active config.
9. Tests at configuration, cache, authenticated OCR HTTP, deployment and real Paperless seams; Python/Rust/Compose checks, browser validation and documented limits.

## Implementation order

Configuration/runtime adapter → persistent cache/diagnostics → HTTP integration → UI/monitoring → presets/proxy/comparison tool → real integration and review.

Runtime settings belong to app-config, secrets/deployment construction to environment. Cache and monitoring are private data, outside source and config history. Keep changes local to OCR whenever possible.

## Verification constraints

Use synthetic committed fixtures. The user's authorized sample/weights may be mounted read-only in local tests but never copied to this public fork. Resource-heavy tests run sequentially. Existing unrelated services remain untouched.

The upstream image uses Python 3.10; ONNX Runtime must support it. Keep the validated upstream stack, add a compatible pinned ONNX dependency, and separately test the actual runtime. A build recipe alone is not evidence of inference or authentication working.
