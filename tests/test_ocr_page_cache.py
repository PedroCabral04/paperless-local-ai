from concurrent.futures import ThreadPoolExecutor
import threading
import time

from ocr_page_cache import PageCache, diagnose


def payload(text='SYNTHETIC EMPLOYEE ALPHA'):
    return {'width': 200, 'height': 100, 'dpi': [150, 150], 'text': text,
            'lines': [{'text': text, 'score': .9, 'poly': [[0,0],[180,0],[180,20],[0,20]]}]}


def test_restart_reuses_completed_page_and_config_change_invalidates(tmp_path):
    image=tmp_path/'page.png';image.write_bytes(b'synthetic raster')
    calls=[]
    def compute():calls.append(1);return payload()
    cache=PageCache(tmp_path/'cache')
    assert cache.get_or_compute(image, {'profile':'one'}, compute)[1] is False
    assert PageCache(tmp_path/'cache').get_or_compute(image, {'profile':'one'}, compute)[1] is True
    assert cache.get_or_compute(image, {'profile':'two'}, compute)[1] is False
    image.write_bytes(b'changed raster')
    assert cache.get_or_compute(image, {'profile':'two'}, compute)[1] is False
    assert len(calls)==3


def test_failure_never_creates_completed_checkpoint(tmp_path):
    import pytest
    image=tmp_path/'page.png';image.write_bytes(b'one')
    cache=PageCache(tmp_path/'cache')
    def fail():raise RuntimeError('synthetic inference failure')
    with pytest.raises(RuntimeError):cache.get_or_compute(image,{},fail)
    assert cache.get_or_compute(image,{},payload)[1] is False


def test_duplicate_concurrent_pages_run_inference_once(tmp_path):
    image=tmp_path/'page.png';image.write_bytes(b'one')
    cache=PageCache(tmp_path/'cache');calls=[];gate=threading.Barrier(2)
    def compute():calls.append(1);time.sleep(.05);return payload()
    def request():gate.wait();return cache.get_or_compute(image,{},compute)[1]
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:request(),range(2)))
    assert sorted(results)==[False,True]
    assert len(calls)==1


def test_budget_and_expiry_are_enforced(tmp_path):
    image=tmp_path/'page.png';image.write_bytes(b'one')
    clock=[1000.]
    cache=PageCache(tmp_path/'cache',clock=lambda:clock[0])
    assert cache.get_or_compute(image,{},payload,ttl_days=1)[1] is False
    clock[0]+=86401
    assert cache.get_or_compute(image,{},payload,ttl_days=1)[1] is False
    assert cache.get_or_compute(image,{},payload,max_bytes=1)[1] is False
    assert cache.status()['entries']==0


def test_corrupt_payload_is_not_trusted(tmp_path):
    import sqlite3
    image=tmp_path/'page.png';image.write_bytes(b'one');cache=PageCache(tmp_path/'cache')
    cache.get_or_compute(image,{},payload)
    with sqlite3.connect(cache.path) as db:db.execute("UPDATE pages SET payload=?",(b'corrupt',))
    assert cache.get_or_compute(image,{},payload)[1] is False


def test_advisory_diagnostics_never_fabricate_accuracy():
    assert 'no_text' in diagnose({'text':'','lines':[]})
    assert 'repeated_lines' in diagnose({'text':'X\n'*8,'lines':[{'text':'X','score':.99}]*8})
    assert 'low_line_confidence' in diagnose({'text':'example','lines':[{'text':'example','score':.2}]})
    assert diagnose(payload())==[]


def test_monitoring_persists_bounded_private_request_metadata(tmp_path):
    cache=PageCache(tmp_path/'cache')
    cache.request_started('request-one','opaque-document',2)
    assert cache.status()['recent'][0]['status']=='running'
    cache.request_finished('request-one',cached=True,elapsed_s=.2,warnings=['no_text'])
    status=PageCache(tmp_path/'cache').status()
    assert status['recent'][0]['status']=='ready'
    assert status['recent'][0]['page']==2
    assert status['requests']['hits']==1
    assert 'text' not in status['recent'][0]
