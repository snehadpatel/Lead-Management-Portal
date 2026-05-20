"""Build SBERT embeddings for a fund catalog CSV/JSON and save a cache pickle.

Usage:
  python scripts/build_fund_embeddings.py --input funds.csv --out models/fund_embeddings.pkl

Expected input CSV columns: scheme_code, scheme_name, category, description
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List

import numpy as np


def load_funds_from_csv(path: Path) -> List[dict]:
    import csv
    funds = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            funds.append({
                'scheme_code': row.get('scheme_code') or row.get('SchemeCode') or row.get('scheme_code', ''),
                'scheme_name': row.get('scheme_name') or row.get('SchemeName') or row.get('scheme_name', ''),
                'category': row.get('category') or row.get('Category') or '',
                'description': row.get('description') or row.get('Description') or ''
            })
    return funds


def build_embeddings(funds: List[dict], model_name: str = 'sentence-transformers/all-mpnet-base-v2') -> dict:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError('sentence-transformers not installed') from e

    model = SentenceTransformer(model_name)
    texts = [ (f.get('scheme_name','') + ' ' + f.get('description','')).strip() for f in funds ]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    return {'funds': funds, 'embeddings': np.asarray(embeddings)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Input CSV or JSON of funds')
    p.add_argument('--out', required=True, help='Output pickle file path')
    p.add_argument('--model', default='sentence-transformers/all-mpnet-base-v2')
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print('Input not found:', inp)
        return

    if inp.suffix.lower() in ('.csv', '.txt'):
        funds = load_funds_from_csv(inp)
    else:
        import json
        funds = json.loads(inp.read_text(encoding='utf-8'))

    data = build_embeddings(funds, model_name=args.model)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'wb') as f:
        pickle.dump(data, f)

    print('✅ Built embeddings and saved to', outp)


if __name__ == '__main__':
    main()
