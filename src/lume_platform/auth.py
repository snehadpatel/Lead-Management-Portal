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


def hash_password(password: str) -> str:
    """Hash password using hashlib PBKDF2 (SHA-256) with a random salt."""
    import hashlib
    import os
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt.hex() + '$' + key.hex()


def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verify password by parsing the salt and checking the PBKDF2 key match."""
    import hashlib
    try:
        salt_hex, key_hex = stored_password.split('$')
        salt = bytes.fromhex(salt_hex)
        key = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return key.hex() == key_hex
    except Exception:
        return False

