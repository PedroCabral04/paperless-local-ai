"""Read-only OCR comparison: isolated model processes, no config/Paperless writes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def child(config_path: Path, output: Path):
    from paddleocr import PaddleOCR
    from ocr_runtime import backend_options, file_digest
    from service import _run_paddle
    config=json.loads(config_path.read_text())
    backend=config['backend']
    started=time.monotonic()
    options={'device':'cpu','text_detection_model_name':f"PP-OCRv6_{config['model_profile']}_det",
             'text_recognition_model_name':f"PP-OCRv6_{config['model_profile']}_rec",
             'use_doc_orientation_classify':False,'use_doc_unwarping':False,'use_textline_orientation':False,
             'cpu_threads':config['cpu_threads'],'enable_hpi':os.getenv('OCR_ENABLE_HPI','false').lower()=='true',
             'enable_mkldnn':True}
    options.update(backend_options(config,config['cpu_threads']))
    model=PaddleOCR(**options)
    load=time.monotonic()-started
    _run_paddle(Path(config['images'][0]),model,config['det_limit_side_len'])
    records=[]
    for image in config['images']:
        timings=[];signatures=[]
        for repeat in range(config['repeats']):
            result=_run_paddle(Path(image),model,config['det_limit_side_len'])
            timings.append(result['performance']['inference_seconds'])
            signatures.append(hashlib.sha256(result['text'].encode()).hexdigest())
            if config['include_text']:
                records.append({'image_sha256':file_digest(Path(image)),'repeat':repeat+1,'text':result['text']})
        records.append({'image_sha256':file_digest(Path(image)),'inference_median_s':statistics.median(timings),
                        'inference_times_s':timings,'text_sha256':signatures,'stable_text':len(set(signatures))==1,
                        'regions':len(result['lines'])})
    output.write_text(json.dumps({'backend':backend,'status':'ready','load_s':load,'records':records,
                                 'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}),encoding='utf8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images',type=Path,nargs='+')
    ap.add_argument('--backends',nargs='+',choices=['paddle','onnxruntime'],default=['paddle','onnxruntime'])
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--det-limit',type=int,default=2304)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--include-text',action='store_true',help='Store private OCR text in output; omit for content hashes only.')
    ap.add_argument('--out',type=Path)
    ap.add_argument('--child-config',type=Path,help=argparse.SUPPRESS)
    args=ap.parse_args()
    if args.child_config:
        try:
            child(args.child_config,args.out)
        except Exception as error:
            args.out.write_text(json.dumps({'status':'failed','error_type':type(error).__name__}),encoding='utf8')
            raise SystemExit(1) from None
        return
    if not args.images or not args.out or not 1<=args.threads<=128 or not 1<=args.repeats<=10 or not 960<=args.det_limit<=4000:
        ap.error('Specify images/out and valid resource limits')
    for image in args.images:
        if not image.is_file():ap.error('Input image missing')
        if image.resolve()==args.out.resolve():ap.error('Output must not replace an input image')
    from ocr_runtime import runtime_signature
    args.out.parent.mkdir(parents=True,exist_ok=True)
    results=[]
    with tempfile.TemporaryDirectory(prefix='plai-compare-') as tmp:
        for backend in args.backends:
            cfg={'backend':backend,'version':'PP-OCRv6','model_profile':'medium','device':'cpu','cpu_threads':args.threads,
                 'det_limit_side_len':args.det_limit,'onnx_cpu_mem_arena':False,'onnx_mem_pattern':False,
                 'images':[str(p.resolve()) for p in args.images],'repeats':args.repeats,'include_text':args.include_text}
            config=Path(tmp)/'config.json';report=Path(tmp)/'result.json'
            config.write_text(json.dumps(cfg),encoding='utf8')
            # Capture logs privately and never persist them: exceptions may include source paths.
            try:
                run=subprocess.run([sys.executable,__file__,'--child-config',str(config),'--out',str(report)],
                                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=7200)
                result=json.loads(report.read_text(encoding='utf8')) if report.exists() else {'status':'failed','exit_code':run.returncode}
            except subprocess.TimeoutExpired:
                result={'status':'failed','error_type':'TimeoutError'}
            result['backend']=backend
            report.unlink(missing_ok=True)
            result['runtime']=runtime_signature({k:v for k,v in cfg.items() if k not in ['images','include_text']})
            results.append(result)
    args.out.write_text(json.dumps({'results':results,'limits':'Selected images, warm inference only; no measured accuracy without human reference.'},indent=2),encoding='utf8')
    args.out.chmod(0o600)
    print(json.dumps({'backends':[{k:r[k] for k in ['backend','status']} for r in results]}))
    if all(r['status']=='failed' for r in results):raise SystemExit(1)


if __name__=='__main__':main()
