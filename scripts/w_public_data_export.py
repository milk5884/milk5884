#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

BASE = 'https://raw.githubusercontent.com/BoatraceCSV/boatracecsv.github.io/main'
OUT = Path('out')
OUT.mkdir(exist_ok=True)
manifest = []

for day in range(9, 16):
    date = f'2026-09-{day:02d}'
    yyyy, mm, dd = date.split('-')
    sources = {
        'race_card': f'data/programs/race_cards/{yyyy}/{mm}/{dd}.csv',
        'result': f'data/results/realtime/{yyyy}/{mm}/{dd}.csv',
        'payout': f'data/results/payouts/{yyyy}/{mm}/{dd}.csv',
    }
    for kind, rel in sources.items():
        url = f'{BASE}/{rel}'
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
        path = OUT / f'{kind}_{date}.csv'
        path.write_bytes(data)
        manifest.append({
            'date': date,
            'kind': kind,
            'url': url,
            'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(),
        })

(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'files': len(manifest), 'bytes': sum(x['bytes'] for x in manifest)}))
