"""Simple JWT auth helpers for FastAPI endpoints.
Uses ADMIN_API_KEY as secret to mint short-lived tokens for admin users.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt

ALGORITHM = os.environ.get('JWT_ALGORITHM', 'HS256')
SECRET_KEY = os.environ.get('JWT_SECRET', os.environ.get('ADMIN_API_KEY', 'dev-secret'))
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get('JWT_EXPIRE_MINUTES', '60'))


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = {"sub": subject}
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
