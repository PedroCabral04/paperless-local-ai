"""Private, bounded, atomic OCR checkpoints. A cache hit never launches a model."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import math
import sqlite3
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from ocr_runtime import file_digest


def diagnose(payload: dict) -> list[str]:
    """Heuristics for review only; these are not calibrated accuracy scores."""
    lines = payload.get('lines', [])
    texts = [' '.join(str(line.get('text', '')).split()) for line in lines]
    nonempty = [text for text in texts if text]
    warnings = []
    if not str(payload.get('text', '')).strip():
        warnings.append('no_text')
    if len(nonempty) >= 6 and Counter(nonempty).most_common(1)[0][1] / len(nonempty) >= .6:
        warnings.append('repeated_lines')
    scores = [float(line['score']) for line in lines if isinstance(line.get('score'), (int, float)) and math.isfinite(line['score'])]
    if scores and sum(scores) / len(scores) < .6:
        warnings.append('low_line_confidence')
    return warnings


def valid_payload(value: dict) -> bool:
    return (isinstance(value, dict) and isinstance(value.get('text'), str)
            and isinstance(value.get('lines'), list)
            and isinstance(value.get('width'), (int, float)) and value['width'] > 0
            and isinstance(value.get('height'), (int, float)) and value['height'] > 0)


class PageCache:
    """SQLite is local-only. 256 lock buckets bound lock-file growth."""

    def __init__(self, directory: Path, *, clock=time.time):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.directory / 'pages.sqlite3'
        self.clock = clock
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS pages (
                    key TEXT PRIMARY KEY, payload BLOB NOT NULL, checksum TEXT NOT NULL,
                    size INTEGER NOT NULL, created REAL NOT NULL, accessed REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY, document TEXT, page INTEGER, status TEXT NOT NULL,
                    started REAL NOT NULL, finished REAL, cached INTEGER NOT NULL DEFAULT 0,
                    elapsed REAL, warnings TEXT NOT NULL DEFAULT '[]', error_type TEXT
                );
                CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
            ''')
        self.path.chmod(0o600)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute('PRAGMA auto_vacuum=FULL')
        # Rollback journal avoids an unbounded WAL alongside the bounded cache.
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA synchronous=FULL')
        return db

    @contextmanager
    def key_lock(self, key: str):
        with (self.directory / (key[:2]+'.lock')).open('a+b') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def prune(self, db, max_bytes: int, ttl_days: int):
        db.execute('DELETE FROM pages WHERE created < ?', (self.clock()-ttl_days*86400,))
        size = db.execute('SELECT COALESCE(SUM(size),0) FROM pages').fetchone()[0]
        while size > max_bytes:
            row = db.execute('SELECT key,size FROM pages ORDER BY accessed LIMIT 1').fetchone()
            if row is None:
                break
            db.execute('DELETE FROM pages WHERE key=?', (row[0],))
            size -= row[1]

    def get_or_compute(self, image: Path, signature: dict, compute, *, max_bytes=512*1024*1024, ttl_days=30):
        key = hashlib.sha256((file_digest(image)+json.dumps(signature,sort_keys=True,separators=(',',':'))).encode()).hexdigest()
        with self.key_lock(key):
            with self.connect() as db:
                self.prune(db,max_bytes,ttl_days)
                row = db.execute('SELECT payload,checksum FROM pages WHERE key=?',(key,)).fetchone()
                if row:
                    try:
                        assert hashlib.sha256(row[0]).hexdigest() == row[1]
                        payload = json.loads(gzip.decompress(row[0]))
                        assert valid_payload(payload)
                    except (AssertionError, ValueError, OSError, EOFError):
                        db.execute('DELETE FROM pages WHERE key=?',(key,))
                    else:
                        db.execute('UPDATE pages SET accessed=? WHERE key=?',(self.clock(),key))
                        return payload, True, key
            payload = compute() # No transaction/checkpoint exists while inference runs.
            if not valid_payload(payload):
                raise ValueError('OCR returned an invalid cache payload')
            encoded = gzip.compress(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode(),mtime=0)
            if len(encoded) <= max_bytes:
                with self.connect() as db:
                    db.execute('INSERT OR REPLACE INTO pages VALUES (?,?,?,?,?,?)',
                               (key,encoded,hashlib.sha256(encoded).hexdigest(),len(encoded),self.clock(),self.clock()))
                    self.prune(db,max_bytes,ttl_days)
            return payload, False, key

    def request_started(self, request_id: str, document: str | None, page: int | None):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO requests (id,document,page,status,started) VALUES (?,?,?,?,?)',
                       (request_id,document,page,'running',self.clock()))
            db.execute('DELETE FROM requests WHERE id NOT IN (SELECT id FROM requests ORDER BY started DESC LIMIT 1000)')

    def request_finished(self, request_id: str, *, cached=False, elapsed_s=0., warnings=(), error_type=None):
        status = 'failed' if error_type else 'ready'
        with self.connect() as db:
            db.execute('UPDATE requests SET status=?,finished=?,cached=?,elapsed=?,warnings=?,error_type=? WHERE id=?',
                       (status,self.clock(),int(cached),elapsed_s,json.dumps(list(warnings)),error_type,request_id))
            key = 'failures' if error_type else ('hits' if cached else 'computed')
            db.execute('INSERT INTO stats VALUES (?,1) ON CONFLICT(key) DO UPDATE SET value=value+1',(key,))

    def status(self):
        with self.connect() as db:
            entries, size = db.execute('SELECT COUNT(*),COALESCE(SUM(size),0) FROM pages').fetchone()
            counters = dict(db.execute('SELECT key,value FROM stats'))
            db.row_factory=sqlite3.Row
            rows=db.execute('SELECT * FROM requests ORDER BY started DESC LIMIT 50').fetchall()
        recent=[]
        for row in rows:
            item=dict(row);item['warnings']=json.loads(item['warnings'])
            if item['status']=='running' and self.clock()-item['started']>3600:
                item['status']='unconfirmed' # Never claim a stale request completed.
            recent.append(item)
        return {'entries':entries,'compressed_bytes':size,'requests':counters,'recent':recent,
                'limits':'Recent received page requests only; not the Paperless queue or full document coverage.'}

    def maintain(self, *, max_bytes=512*1024*1024, ttl_days=30):
        with self.connect() as db:
            self.prune(db,max_bytes,ttl_days)
