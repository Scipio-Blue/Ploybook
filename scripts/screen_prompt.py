"""Inspect a user-supplied request with explicit opt-in to hosted inference."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt', required=True)
    p.add_argument('--allow-api', action='store_true')
    p.add_argument('--model-path', default='Qwen/Qwen2.5-14B-Instruct')
    p.add_argument('--encoder-path', default='facebook/contriever-msmarco')
    p.add_argument('--device', default='cuda')
    p.add_argument('--local-files-only', action='store_true')
    a = p.parse_args()
    if not a.allow_api: p.error('Hosted inference requires --allow-api and your own JEV_KEY.')
    if not os.environ.get('JEV_KEY'): p.error('Set JEV_KEY in your environment.')
    from ploybook import Ploybook
    detector = Ploybook(a.model_path, a.encoder_path, a.device, a.local_files_only)
    print(json.dumps(detector.inspect(a.prompt), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
