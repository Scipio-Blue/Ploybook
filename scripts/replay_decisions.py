"""Reproduce all primary verdicts and the post-hoc 25-policy grid, offline."""
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['PLOYBOOK_OFFLINE'] = '1'
sys.path.insert(0, str(ROOT / 'src'))
from ploybook import score_policy


def main():
    rows = list(csv.DictReader((ROOT / 'decision-scores.csv').open()))
    original = {r['sample_id']:r for r in csv.DictReader(
        (ROOT / 'results/full/exp_full.csv').open()) if r['arm'] == 'unmask'}
    assert len(rows) == len(original) == 1294
    for r in rows:
        a = score_policy(float(r['core']), float(r['severity']), int(r['fast_block']))
        b = original[r['sample_id']]
        assert a['blocked'] == bool(int(b['blocked']))
        assert abs(a['p_harmful'] - float(b['p_harmful'])) < 1e-8
    grid = []
    for lower in [.25,.30,.35,.40,.45]:
        for severe in [.60,.65,.70,.75,.80]:
            counts = {'harmful':0,'benign':0,'pair':0}
            for r in rows:
                b = int(score_policy(float(r['core']), float(r['severity']),
                    int(r['fast_block']), lower, severe)['blocked'])
                counts['benign' if r['family']=='benign' else 'harmful'] += b
                if r['family']=='pair': counts['pair'] += b
            grid.append({'lower':lower,'severity_cutoff':severe,**counts})
    expected = list(csv.DictReader((ROOT / 'threshold-grid.csv').open()))
    assert len(expected) == len(grid) == 25
    for e,g in zip(expected,grid):
        assert all(abs(float(e[k]) - v) < 1e-8 for k,v in g.items())
    out = ROOT / 'reproduced'; out.mkdir(exist_ok=True)
    (out / 'threshold-grid.json').write_text(json.dumps(grid, indent=2)+'\n')
    print('PASS: 1,294 exact score/verdict matches; 25 policies reproduced; zero network calls.')


if __name__ == '__main__': main()
