#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import date, timedelta
from pathlib import Path

BASE = 'https://raw.githubusercontent.com/BoatraceCSV/boatracecsv.github.io/main'
START = date(2026, 7, 19)
END = date(2026, 9, 8)
OUT = Path('out')
OUT.mkdir(exist_ok=True)
manifest = []

d = START
while d <= END:
    ds = d.isoformat()
    yyyy, mm, dd = ds.split('-')
    sources = {
        'od3': f'data/previews/od3/{yyyy}/{mm}/{dd}.csv',
        'payout': f'data/results/payouts/{yyyy}/{mm}/{dd}.csv',
    }
    for kind, rel in sources.items():
        url = f'{BASE}/{rel}'
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
        except Exception as exc:
            manifest.append({'date': ds, 'kind': kind, 'url': url, 'error': repr(exc)})
            continue
        path = OUT / f'{kind}_{ds}.csv'
        path.write_bytes(data)
        manifest.append({
            'date': ds,
            'kind': kind,
            'url': url,
            'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(),
        })
    d += timedelta(days=1)

(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps({
    'start': START.isoformat(), 'end': END.isoformat(),
    'records': len(manifest),
    'files': sum('bytes' in x for x in manifest),
    'errors': sum('error' in x for x in manifest),
    'bytes': sum(x.get('bytes', 0) for x in manifest),
}))
