import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


def test_password_uses_utf8_and_lf_without_windows_text_translation(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/setup-admin-auth.py'
    spec=importlib.util.spec_from_file_location('admin_setup',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    password='synthetic-password-for-test'
    monkeypatch.setattr(module.getpass,'getpass',lambda _:password)
    monkeypatch.setattr(sys,'argv',[str(path),'--data-dir',str(tmp_path)])
    seen={}
    def hash_password(args,**kwargs):
        seen.update(kwargs)
        assert password not in args
        return SimpleNamespace(stdout=('$2a$14$'+'A'*53).encode())
    monkeypatch.setattr(module.subprocess,'run',hash_password)
    module.main()
    assert seen['input']==password.encode()+b'\n'
    assert not seen.get('text',False)
    assert password not in (tmp_path/'secrets/plai-admin-auth').read_text()
