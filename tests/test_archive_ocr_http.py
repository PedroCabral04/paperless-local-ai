import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import service
from app_config import validate_config


def test_authenticated_http_reuses_page_without_launching_worker(tmp_path,monkeypatch):
    cfg=validate_config({'ocr':{'backend':'onnxruntime'}})
    monkeypatch.setattr(service,'TOKEN','synthetic-token')
    monkeypatch.setattr(service,'TMP_DIR',tmp_path)
    monkeypatch.setattr(service,'CACHE_DIR',tmp_path/'cache')
    monkeypatch.setattr(service,'load_app_config',lambda:cfg)
    for name in ['write_recovery_state','set_idle_state','consume_retry_now']:
        monkeypatch.setattr(service,name,lambda *args,**kwargs:None)
    monkeypatch.setattr(service,'read_recovery_state',lambda:{})
    class FakeSession:
        calls=0
        def note_request(self):pass
        def ocr(self,path,*,config=None):
            self.calls+=1
            assert config['backend']=='onnxruntime'
            return {'text':'SYNTHETIC ALPHA','width':200,'height':100,'lines':[{'text':'SYNTHETIC ALPHA','score':.9,'poly':[[0,0],[180,0],[180,20],[0,20]]}]}
    session=FakeSession();monkeypatch.setattr(service,'_session',lambda:session)
    server=ThreadingHTTPServer(('127.0.0.1',0),service.Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        for endpoint in ['/v1/status','/v1/ocr']:
            req=Request(base+endpoint,data=b'image' if endpoint.endswith('ocr') else None)
            try:urlopen(req)
            except HTTPError as exc:assert exc.code==401
            else:raise AssertionError('Private endpoint must require token')
        def post():
            req=Request(base+'/v1/ocr',data=b'synthetic-raster',headers={'Authorization':'Bearer synthetic-token','X-PLAI-Page-Number':'1'})
            with urlopen(req) as response:return json.load(response)
        assert post()['cache']['hit'] is False
        assert post()['cache']['hit'] is True
        assert session.calls==1
        req=Request(base+'/v1/status',headers={'Authorization':'Bearer synthetic-token'})
        with urlopen(req) as response:status=json.load(response)
        assert status['requests']['hits']==1
        assert all('text' not in row for row in status['recent'])
    finally:
        server.shutdown();server.server_close();thread.join()
