#!/usr/bin/env python3
"""Explicit synthetic Paperless integration test; leaves its labelled test document."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import uuid

import requests


def synthetic_scan(path):
    from PIL import Image,ImageDraw,ImageFont
    from pypdf import PdfReader,PdfWriter
    from pypdf.generic import DictionaryObject,NameObject,DecodedStreamObject
    image=Image.new('RGB',(1200,1600),'white')
    draw=ImageDraw.Draw(image)
    draw.text((90,160),'SYNTHETIC EMPLOYEE ALPHA',font=ImageFont.load_default(size=48),fill='black')
    draw.text((90,250),'ARCHIVE RECORD 1981',font=ImageFont.load_default(size=40),fill='black')
    memory=io.BytesIO();image.save(memory,format='PDF',resolution=150);memory.seek(0)
    writer=PdfWriter();writer.add_page(PdfReader(memory).pages[0]);page=writer.pages[0]
    font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
    resources=page['/Resources'];resources[NameObject('/Font')]=DictionaryObject({NameObject('/BadOCR'):writer._add_object(font)})
    stream=DecodedStreamObject();stream.set_data(b'q BT /BadOCR 12 Tf 3 Tr 30 40 Td (WRONGOLDLAYER991) Tj ET Q')
    from pypdf.generic import ArrayObject
    contents=page.get('/Contents')
    page[NameObject('/Contents')]=ArrayObject([contents,writer._add_object(stream)])
    with path.open('wb') as output:writer.write(output)
    assert 'WRONGOLDLAYER991' in PdfReader(path).pages[0].extract_text()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paperless-url',required=True)
    ap.add_argument('--work-dir',type=Path,required=True)
    ap.add_argument('--allow-upload',action='store_true')
    args=ap.parse_args()
    if not args.allow_upload:ap.error('Pass --allow-upload to create the synthetic test document')
    token=os.environ.get('TEST_PAPERLESS_TOKEN')
    if not token:ap.error('Set TEST_PAPERLESS_TOKEN in the environment')
    args.work_dir.mkdir(parents=True,exist_ok=True)
    source=args.work_dir/'synthetic-old-ocr.pdf';synthetic_scan(source)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    base=args.paperless_url.rstrip('/');session=requests.Session();session.trust_env=False
    session.headers['Authorization']='Token '+token
    title='SYNTHETIC ARCHIVE OCR TEST '+uuid.uuid4().hex[:12]
    with source.open('rb') as f:
        response=session.post(base+'/api/documents/post_document/',files={'document':(source.name,f,'application/pdf')},data={'title':title},timeout=120)
    response.raise_for_status()
    doc=None;deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        response=session.get(base+'/api/documents/',params={'title__iexact':title},timeout=30);response.raise_for_status()
        doc=next((d for d in response.json()['results'] if d['title']==title),None)
        if doc:break
        time.sleep(2)
    if doc is None:raise TimeoutError('Synthetic consumption did not finish')
    did=doc['id'];content=doc['content']
    assert 'SYNTHETIC EMPLOYEE ALPHA' in content.upper(), 'New OCR text is missing'
    assert 'WRONGOLDLAYER991' not in content, 'Old hidden OCR survived'
    assert doc.get('archived_file_name'), 'No searchable archive was produced'
    original=session.get(base+f'/api/documents/{did}/download/',params={'original':'true'},timeout=90)
    original.raise_for_status();assert hashlib.sha256(original.content).hexdigest()==source_hash
    # Deliberately alter only this synthetic document, proving native reprocess
    # restores content from the OCR path rather than preserving a manual PATCH.
    response=session.patch(base+f'/api/documents/{did}/',json={'content':'temporary synthetic reprocess marker'},timeout=30);response.raise_for_status()
    response=session.post(base+'/api/documents/reprocess/',json={'documents':[did],'remote_ocr':False},timeout=30);response.raise_for_status()
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        response=session.get(base+f'/api/documents/{did}/',timeout=30);response.raise_for_status()
        if response.json()['content']==content:break
        time.sleep(2)
    else:raise TimeoutError('Native reprocess did not restore new OCR text')
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        response=session.get(base+'/api/documents/',params={'query':'content:"SYNTHETIC EMPLOYEE ALPHA"'},timeout=30);response.raise_for_status()
        if any(d['id']==did for d in response.json()['results']):break
        time.sleep(2)
    else:raise TimeoutError('Synthetic OCR is not searchable')
    report={'document_id':did,'old_ocr_replaced':True,'new_text_searchable':True,'archive_created':True,
            'original_hash_unchanged':True,'native_reprocess_restored_content':True,'source_sha256':source_hash}
    (args.work_dir/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report))


if __name__=='__main__':main()
