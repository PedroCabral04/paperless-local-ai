#!/usr/bin/env python3
"""Create a local private Caddy password-hash file; never print/store plaintext."""
import argparse
import getpass
import os
from pathlib import Path
import re
import subprocess


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--user',default='operator')
    ap.add_argument('--data-dir',type=Path,default=Path('data'))
    args=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',args.user):ap.error('Use an alphanumeric admin username')
    password=getpass.getpass('Admin password (12+ characters): ')
    if len(password)<12:ap.error('Password must contain at least 12 characters')
    if password!=getpass.getpass('Confirm password: '):ap.error('Passwords differ')
    run=subprocess.run(['docker','run','--rm','-i','caddy:2.11.4-alpine','caddy','hash-password'],
                       input=password+'\n',capture_output=True,text=True,check=True)
    digest=run.stdout.strip()
    if not digest.startswith('$2'):raise RuntimeError('Caddy did not return a bcrypt hash')
    directory=args.data_dir/'secrets';directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    path=directory/'plai-admin-auth'
    # Refuse accidental replacement. Rotation is an explicit operator action.
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w',encoding='utf8') as stream:stream.write(args.user+' '+digest+'\n')
    print('Admin hash created. Start Compose with compose.auth.yaml; use the configured exact admin origin.')


if __name__=='__main__':main()
