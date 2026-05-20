"""Simple recommendation engine that personalizes mutual fund suggestions
based on user profile, inferred risk appetite, and live market context.

This is a prototype: it uses SBERT semantic search when available and
falls back to category-based heuristics. Returns recommended funds with
reasoning and confidence scores.
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from lume_platform.config import MODELS_DIR, EXPORT_DIR, PROJECT_ROOT


class SimpleRecommender:
    def __init__(self, registry=None):
        self.registry = registry
        self.fallback_funds = self._load_funds_catalog()

        # profiles storage
        self.profiles_path = EXPORT_DIR / "users_profiles.json"
        if not self.profiles_path.parent.exists():
            os.makedirs(self.profiles_path.parent, exist_ok=True)

    def save_profile(self, user_id: str, profile: dict) -> None:
        data = {}
        if self.profiles_path.is_file():
            try:
                with open(self.profiles_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                data = {}

        data[str(user_id)] = profile
        with open(self.profiles_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load_profile(self, user_id: str) -> Optional[dict]:
        if not self.profiles_path.is_file():
            return None
        try:
            with open(self.profiles_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get(str(user_id))
        except Exception:
            return None

    def _load_funds_catalog(self) -> List[Dict[str, Any]]:
        catalog_path = MODELS_DIR / "fund_catalog.json"
        if catalog_path.is_file():
            try:
                with open(catalog_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []

        data_path = PROJECT_ROOT / 'data' / 'Mutual_Fund_Data-main' / 'mutual_fund_data.csv'
        if not data_path.is_file():
            return []

        funds: List[Dict[str, Any]] = []
        seen: set[str] = set()
        with open(data_path, 'r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                scheme_name = (row.get('Scheme_Name') or '').strip()
                scheme_code = (row.get('Scheme_Code') or '').strip()
                if not scheme_name or scheme_name in seen:
                    continue
                seen.add(scheme_name)
                raw_category = (row.get('Scheme_Category') or '').strip()
                normalized_category = self._normalize_category(raw_category)
                aum = self._to_float(row.get('Average_AUM_Cr'))
                nav = self._to_float(row.get('NAV'))
                launch_year = self._launch_year(row.get('Launch_Date'))
                return_profile = self._return_profile(normalized_category, aum, nav)
                funds.append({
                    'scheme_code': scheme_code or f"MF-{len(funds) + 1}",
                    'scheme_name': scheme_name,
                    'name': scheme_name,
                    'category': raw_category or normalized_category.title(),
                    'category_group': normalized_category,
                    'amc': (row.get('AMC') or '').strip(),
                    'nav': nav,
                    'aum': aum,
                    'min_investment': self._parse_amount(row.get('Scheme_Min_Amt')),
                    'launch_date': row.get('Launch_Date'),
                    'risk': self._risk_from_category(normalized_category),
                    'rating': self._rating_from_category(normalized_category, aum),
                    'personas': self._personas_from_category(normalized_category),
                    'r1': return_profile['r1'],
                    'r3': return_profile['r3'],
                    'r5': return_profile['r5'],
                    'match_score': return_profile['match_score'],
                    'description': f"{scheme_name} from {row.get('AMC', 'the AMC').strip()} · launched {launch_year or 'unknown'}",
                })

        return funds

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value in (None, '', '-'):
                return default
            return float(str(value).replace(',', '').strip())
        except Exception:
            return default

    def _parse_amount(self, value: Any) -> float:
        try:
            text = str(value or '').replace(',', '').lower()
            number = re.findall(r"[0-9.]+", text)
            return float(number[0]) if number else 0.0
        except Exception:
            return 0.0

    def _launch_year(self, value: Any) -> Optional[int]:
        try:
            text = str(value or '')
            return int(text[:4]) if len(text) >= 4 else None
        except Exception:
            return None

    def _normalize_category(self, category: str) -> str:
        text = (category or '').lower()
        if any(word in text for word in ['liquid', 'overnight', 'money market', 'short duration', 'ultra short', 'corporate bond', 'gilt']):
            return 'conservative'
        if any(word in text for word in ['index', 'nifty', 'etf', 'passive']):
            return 'passive'
        if any(word in text for word in ['balanced', 'hybrid', 'multi asset', 'dynamic asset']):
            return 'balanced'
        if any(word in text for word in ['mid cap', 'small cap', 'sectoral', 'thematic', 'flexi', 'large & mid']):
            return 'growth'
        return 'balanced'

    def _risk_from_category(self, normalized_category: str) -> str:
        return {
            'conservative': 'low',
            'passive': 'medium',
            'balanced': 'medium',
            'growth': 'high',
        }.get(normalized_category, 'medium')

    def _rating_from_category(self, normalized_category: str, aum: float) -> int:
        base = {'conservative': 4, 'passive': 4, 'balanced': 4, 'growth': 5}.get(normalized_category, 3)
        if aum >= 5000:
            base += 1
        return max(3, min(base, 5))

    def _personas_from_category(self, normalized_category: str) -> List[str]:
        return {
            'conservative': ['conservative'],
            'passive': ['passive'],
            'balanced': ['balanced'],
            'growth': ['growth'],
        }.get(normalized_category, ['balanced'])

    def _return_profile(self, normalized_category: str, aum: float, nav: float) -> Dict[str, float]:
        base = {
            'conservative': (6.6, 6.2, 5.8),
            'passive': (11.8, 13.2, 12.6),
            'balanced': (15.4, 14.3, 13.5),
            'growth': (24.8, 19.5, 17.2),
        }.get(normalized_category, (12.0, 11.0, 10.0))
        liquidity_bonus = 0.5 if aum >= 1000 else 0.0
        nav_bonus = min(max(nav / 1000.0, 0.0), 2.0)
        score = 0.45 if normalized_category in {'balanced', 'passive'} else 0.55
        return {
            'r1': round(base[0] + liquidity_bonus + nav_bonus * 0.15, 1),
            'r3': round(base[1] + liquidity_bonus * 0.8 + nav_bonus * 0.12, 1),
            'r5': round(base[2] + liquidity_bonus * 0.6 + nav_bonus * 0.10, 1),
            'match_score': round(score, 4),
        }

    def catalog_view(self, top_k: int = 100, persona: Optional[str] = None, query: Optional[str] = None) -> List[Dict[str, Any]]:
        funds = list(self.fallback_funds or [])
        persona_key = (persona or '').lower().strip()
        query_text = (query or '').lower().strip()

        if persona_key:
            allowed_groups = {
                'growth': {'growth'},
                'balanced': {'balanced', 'passive'},
                'passive': {'passive', 'balanced'},
                'conservative': {'conservative'},
            }.get(persona_key, set())
            if allowed_groups:
                funds = [f for f in funds if f.get('category_group') in allowed_groups or f.get('risk') in ({'low'} if persona_key == 'conservative' else {'medium', 'high'})]

        if query_text:
            tokens = [token for token in re.split(r'[^a-z0-9]+', query_text) if len(token) > 2]
            if tokens:
                funds = [
                    f for f in funds
                    if any(
                        token in f.get('scheme_name', '').lower()
                        or token in f.get('category', '').lower()
                        or token in f.get('amc', '').lower()
                        for token in tokens
                    )
                ]

        def sort_key(fund: Dict[str, Any]) -> tuple:
            score = float(fund.get('match_score', 0.0))
            aum = float(fund.get('aum', 0.0))
            return (score, aum)

        return sorted(funds, key=sort_key, reverse=True)[:top_k]

    def recommend_for_profile(self, profile: dict, market_context: Optional[dict] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return a list of recommended funds with explanations.

        Strategy:
        - If SBERT semantic search is available on registry, build a query combining
          inferred risk profile, investment goals, and market sentiment.
        - Else, fall back to category mapping by `inferred_risk_profile`.
        """
        if market_context is None:
            market_context = {}

        risk = profile.get('inferred_risk_profile', '').lower()
        goals = profile.get('goals', '') or profile.get('investment_objective', '') or ''
        horizon = profile.get('investment_horizon', '')
        market_sent = market_context.get('market_sentiment', '')

        # Build query for semantic search
        query_parts = []
        if risk:
            query_parts.append(risk)
        if goals:
            query_parts.append(goals)
        if horizon:
            query_parts.append(str(horizon))
        if market_sent:
            query_parts.append(market_sent)
        query = " ".join(query_parts).strip() or "mutual funds recommended"

        results = []
        # Prefer SBERT search
        try:
            if self.registry and getattr(self.registry, 'sbert_search', None):
                funds = self.registry.sbert_search.query(query, top_k=top_k)
                for f in funds:
                    # confidence adjusted by market context (simple heuristic)
                    conf = f.get('match_score', 0.5)
                    if market_sent and 'bear' in market_sent.lower() and 'equity' in (f.get('category','').lower()):
                        conf *= 0.75
                    results.append({
                        'scheme_code': f['scheme_code'],
                        'scheme_name': f['scheme_name'],
                        'category': f['category'],
                        'match_score': round(float(conf), 4),
                        'reason': f"Matched by semantic similarity to profile: '{query}'"
                    })
                if results:
                    return results
        except Exception:
            results = []

        # Fallback: category-based recommendations from fallback_funds
        persona_key = 'balanced'
        if 'aggressive' in risk or 'equity' in risk or 'growth' in risk:
            persona_key = 'growth'
        elif 'conservative' in risk or 'debt' in risk or 'liquid' in risk:
            persona_key = 'conservative'
        elif 'passive' in risk or 'index' in risk:
            persona_key = 'passive'

        candidates = self.catalog_view(top_k=top_k * 3, persona=persona_key, query=query)

        for rank, f in enumerate(candidates[:top_k], start=1):
            allocation = max(8, min(40, 24 - (rank - 1) * 4))
            results.append({
                'scheme_code': f.get('scheme_code', 'UNK'),
                'scheme_name': f.get('scheme_name', f.get('name', 'Unknown Fund')),
                'category': f.get('category', 'Unknown'),
                'risk': f.get('risk', 'medium'),
                'aum': f.get('aum', 0.0),
                'match_score': round(float(f.get('match_score', 0.6)), 4),
                'recommended_allocation_pct': allocation,
                'reason': f"Persona fit for '{profile.get('inferred_risk_profile', persona_key)}' based on category {f.get('category_group', f.get('category', 'balanced'))} and live catalog data"
            })

        return results
