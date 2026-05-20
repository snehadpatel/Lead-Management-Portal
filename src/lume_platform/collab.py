"""Basic collaborative filtering utilities using transaction history.

This module constructs a user->fund interaction matrix from a CSV of transactions
and computes simple item scores for a given user based on co-occurrence.
"""
from __future__ import annotations

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

DATASETS_DIR = Path(os.environ.get('DATASETS_DIR', 'datasets'))
TXN_PATH = DATASETS_DIR / 'kaggle' / 'financial_user_behaviors' / 'transactions_data.csv'


class CollabEngine:
    def __init__(self, txn_path: Path | None = None):
        self.txn_path = Path(txn_path) if txn_path else TXN_PATH
        self.user_item = None
        self.items = []

    def build_matrix(self, max_users: int = 10000):
        if not self.txn_path.exists():
            return False
        # Read transactions and build user-fund interactions (binary)
        df = pd.read_csv(self.txn_path)
        # Expect columns: client_id, scheme_code, amount
        if 'client_id' not in df.columns or 'scheme_code' not in df.columns:
            return False
        # Use amount threshold to consider interaction
        df['amount'] = pd.to_numeric(df.get('amount', 0), errors='coerce').fillna(0)
        df = df[df['amount'] > 0]

        # Pivot to user-item
        pivot = pd.pivot_table(df, index='client_id', columns='scheme_code', values='amount', aggfunc='sum', fill_value=0)
        # Binarize
        pivot = (pivot > 0).astype(int)

        # Optionally limit users
        if pivot.shape[0] > max_users:
            pivot = pivot.sample(n=max_users, random_state=42)

        self.user_item = pivot
        self.items = list(pivot.columns)
        return True

    def recommend_for_user(self, user_id: str, top_k: int = 5) -> List[Dict]:
        if self.user_item is None:
            ok = self.build_matrix()
            if not ok:
                return []

        if str(user_id) not in map(str, self.user_item.index):
            # Cold start: recommend top popular items
            popular = self.user_item.sum(axis=0).sort_values(ascending=False)
            return [{'scheme_code': idx, 'score': float(popular.loc[idx])} for idx in popular.index[:top_k]]

        user_vec = self.user_item.loc[str(user_id)].values
        # Item-item similarity: co-occurrence matrix
        item_matrix = self.user_item.values.T  # items x users
        coocc = item_matrix.dot(item_matrix.T)
        # Score candidate items by similarity to user's items
        item_scores = coocc.dot(user_vec)
        # Map back to scheme codes
        idxs = np.argsort(-item_scores)[:top_k+10]
        results = []
        for i in idxs:
            code = self.items[int(i)]
            if code in self.user_item.columns and self.user_item.loc[str(user_id), code] == 1:
                continue
            results.append({'scheme_code': code, 'score': float(item_scores[i])})
            if len(results) >= top_k:
                break
        return results
