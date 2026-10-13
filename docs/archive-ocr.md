# Archive OCR fork

This branch extends upstream paperless-local-ai with an explicit ONNX Runtime CPU backend, a persistent page-result cache, resource controls and an optional authenticated admin proxy. Paperless remains responsible for documents, permissions and text search. Existing Paddle deployments retain their defaults.

## Install from this branch

Use the checkout's `compose.dev.yaml` to build both application images. The upstream published images do **not** contain these fork changes. Configure `.env` using the upstream installation instructions, including separate Paperless and OCR service tokens. Keep secrets out of app-config and Git.

Generate the admin hash interactively:

```sh
python scripts/setup-admin-auth.py --user operator
docker compose -f compose.yaml -f compose.dev.yaml -f compose.auth.yaml up -d --build
```

Set `--data-dir` to the same directory as `APP_DATA_DIR` when overriding the default. Open exactly `http://localhost:30148`, or configure `PLAI_ADMIN_ORIGIN` and `PLAI_ADMIN_PORT` together. The optional proxy protects HTML and admin APIs and removes direct core host ports. Use TLS at the appropriate trusted gateway before exposing it beyond localhost. The existing Paperless RAG relay retains its own secret authentication. Without `compose.auth.yaml`, the upstream Control Center remains unauthenticated.

For password rotation, stop the proxy, move the old hash file into protected backup storage, rerun the helper and restart the proxy. Windows operators should restrict the data directory ACL; POSIX file modes alone do not configure Windows ACLs.

## Historical scan preset

`deploy/archive/app-config.example.json` contains only the OCR section. Merge that section into your existing app-config, or enter those values in Control Center → App Settings. Do not overwrite connection or metadata settings with the partial example. ONNX currently supports CPU PP-OCRv6 Medium only. Resource limits in Compose must agree with the selected thread count; six model threads do not override a four-CPU container quota.

Adapt `deploy/archive/paperless.override.example.yaml` to your existing Paperless service, shared integration mount and network. Its `redo` setting replaces existing OCR when Paperless processes/reprocesses a document. It does not automatically enqueue the existing archive. Test a small selection first, then use Paperless's reprocess action in manageable batches. The uploaded original remains unchanged; searchable text belongs to Paperless content and the archive PDF.

The external plugin still needs Paperless's configured Tesseract language installed for startup validation (`PAPERLESS_OCR_LANGUAGES=por`). Tesseract is not the recognition engine selected by this preset. Layout detection is not required for native text search and is not added to this path.

## Cache and monitoring

Page results live under `/ocr-data/page-cache`, outside configuration history. **They contain private OCR text and coordinates.** Protect/back up/delete this storage like the archive itself. Keys include image content, effective OCR settings, model artifacts, package versions and relevant runtime source. Completed inference can be reused after a restart; rasterization and PDF assembly may still repeat. Failures are not cached.

The limit covers compressed cached payload, not an exact total directory quota: SQLite metadata, request history and lock files add overhead. Expiry and eviction run on requests/status reads. Disabling cache stops reuse; it does not erase stored text. Remove the cache directory only while the OCR service is stopped to purge it completely.

Monitoring reports received OCR requests, cache hits and advisory blank/repeated-line/low-confidence flags. It cannot know documents not yet submitted or how many pages remain in a PDF. Confidence and absence of warnings do not prove transcription accuracy. Native search can miss names that OCR omitted or changed; validate important searches against scans.

## Compare engines locally

Mount selected page images read-only and a private output directory into the OCR container, then run:

```sh
python /app/compare_ocr.py --images /inputs/page-1.png /inputs/page-2.png \
  --backends paddle onnxruntime --threads 4 --det-limit 2304 --repeats 3 \
  --out /private/comparison.json
```

The command uses separate model processes, one warmup page, warm inference timings and process peak RSS. It uses the shared AI lock and does not change active configuration or Paperless. Default output contains text hashes; `--include-text` deliberately writes private OCR text for human evaluation. Never upload that output to a public repository. ONNX uses pinned, checksum-verified model exports. Native Paddle uses its cached Paddle-format weights; report provenance when comparing them.

`--backends openvino` is experimental and requires the matching UltraInfer/HPI wheel in a separate test environment. It explicitly selects OpenVINO through PaddleX model configuration; installation or inference failure is recorded as failure, not silently labelled as another engine. This option is not a supported Control Center backend.

## Verification and limits

`scripts/verify-archive-ocr.py --help` describes the opt-in synthetic Paperless integration check. It uploads a clearly labelled synthetic scan with an intentionally wrong old text layer and verifies replacement, native reprocess, content search, archive creation and original SHA-256 preservation. It leaves the test document for audit. The local fork check used Paperless 3.3.0 / OCRmyPDF 17.12.1; this is not a claim of every deployment or certified archival conformance.

Use local processing for private archives unless their owner authorizes another environment. This fork does not require a cloud subscription, GPU or Ollama for OCR-only use. At large archive sizes, retained page caches and archive PDFs require additional disk space beyond source PDFs. Start with representative scans and measure throughput and name recovery before planning the full batch.
