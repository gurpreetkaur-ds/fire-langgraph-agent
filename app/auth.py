"""Accounts: sign-up (gated by an invite code), sign-in, cookie sessions, and the request guards.

- Passwords: scrypt (stdlib), per-user random salt, constant-time comparison.
- Sessions: a random 256-bit token in an HttpOnly, SameSite=Lax cookie; only its SHA-256 is stored.
- CSRF: every state-changing /api route also requires the `X-Requested-With: fetch` header. Browsers cannot send
  custom headers cross-site without a CORS preflight (which this app never grants), and SameSite=Lax already keeps
  the cookie off cross-site POSTs, so the two together block CSRF.
- Invite code: when AGENT_ACCESS_TOKEN is set (production), sign-up requires it, so strangers cannot create
  accounts and spend Fire credit.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request, Response
from sqlalchemy import delete, func, select

from evaluation.db import session_scope
from evaluation.models import utcnow

from .models import User, UserSession
from .ratelimit import limiter
from .settings import get_app_settings

COOKIE_NAME = "agent_session"
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")
MIN_PASSWORD, MAX_PASSWORD = 10, 200
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}


@dataclass(frozen=True)
class CurrentUser:
    id: str
    username: str
    display_name: str
    preferences: dict

    def public(self) -> dict:
        return {"id": self.id, "username": self.username, "display_name": self.display_name, "preferences": self.preferences}


# ---- passwords -------------------------------------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    return "scrypt${n}${r}${p}${salt}${digest}".format(
        **_SCRYPT, salt=base64.b64encode(salt).decode(), digest=base64.b64encode(digest).decode()
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if algo != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt_b64), dklen=32, n=int(n), r=int(r), p=int(p)
        )
        return hmac.compare_digest(digest, base64.b64decode(digest_b64))
    except (ValueError, TypeError):
        return False


_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))  # equalises timing when the username does not exist


# ---- sessions --------------------------------------------------------------------------------------------

def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def start_session(response: Response, user_id: str) -> None:
    settings = get_app_settings()
    token = secrets.token_urlsafe(32)
    now = utcnow()
    with session_scope() as db:
        db.add(UserSession(token_hash=_token_hash(token), user_id=user_id, created_at=now,
                           expires_at=now + timedelta(days=settings.session_days)))
        db.execute(delete(UserSession).where(UserSession.expires_at < now))  # housekeeping
        user = db.get(User, user_id)
        if user is not None:
            user.last_login_at = now
    response.set_cookie(
        COOKIE_NAME, token, max_age=settings.session_days * 86400, httponly=True, samesite="lax",
        secure=settings.cookie_secure, path="/",
    )


def end_session(request: Request, response: Response) -> None:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        with session_scope() as db:
            db.execute(delete(UserSession).where(UserSession.token_hash == _token_hash(token)))
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, samesite="lax", secure=get_app_settings().cookie_secure)


def user_from_request(request: Request) -> Optional[CurrentUser]:
    token = request.cookies.get(COOKIE_NAME)
    if not token or len(token) > 200:
        return None
    with session_scope() as db:
        sess = db.get(UserSession, _token_hash(token))
        if sess is None:
            return None
        expires = sess.expires_at if sess.expires_at.tzinfo else sess.expires_at.replace(tzinfo=utcnow().tzinfo)
        if expires < utcnow():
            db.delete(sess)
            return None
        user = db.get(User, sess.user_id)
        if user is None:
            return None
        return CurrentUser(user.id, user.username, user.display_name, dict(user.preferences or {}))


# ---- guards ----------------------------------------------------------------------------------------------

def current_user(request: Request) -> CurrentUser:
    user = user_from_request(request)
    if user is None:
        raise HTTPException(401, "Please sign in.")
    return user


def require_csrf(request: Request) -> None:
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    if request.headers.get("x-requested-with") != "fetch":
        raise HTTPException(403, "Missing X-Requested-With header.")


def write_user(request: Request, user: CurrentUser = Depends(current_user)) -> CurrentUser:
    """For state-changing routes: signed in + CSRF header."""
    require_csrf(request)
    return user


# ---- account operations ----------------------------------------------------------------------------------

def invite_required() -> bool:
    return bool(os.environ.get("AGENT_ACCESS_TOKEN"))


def _check_invite(code: Optional[str]) -> None:
    expected = os.environ.get("AGENT_ACCESS_TOKEN")
    if not expected:
        return
    if not code or not hmac.compare_digest(code.strip().encode(), expected.encode()):
        raise HTTPException(403, "Wrong invite code.")


def validate_credentials(username: str, password: str) -> str:
    username = (username or "").strip().lower()
    if not USERNAME_RE.match(username):
        raise HTTPException(422, "Username must be 3-32 characters: lowercase letters, digits, '.', '_' or '-'.")
    if not (MIN_PASSWORD <= len(password or "") <= MAX_PASSWORD):
        raise HTTPException(422, f"Password must be {MIN_PASSWORD}-{MAX_PASSWORD} characters.")
    return username


def signup(request: Request, username: str, password: str, display_name: str, invite_code: Optional[str]) -> CurrentUser:
    if not get_app_settings().allow_signup:
        raise HTTPException(403, "Sign-up is closed on this server.")
    limiter.check(f"signup:{_client_ip(request)}", 10, 3600, message="Too many sign-up attempts. Try again later.")
    _check_invite(invite_code)
    username = validate_credentials(username, password)
    display_name = (display_name or "").strip()[:80] or username
    with session_scope() as db:
        if db.scalar(select(func.count()).select_from(User).where(User.username == username)):
            raise HTTPException(409, "That username is taken.")
        user = User(username=username, display_name=display_name, password_hash=hash_password(password), preferences={}, created_at=utcnow())
        db.add(user)
        db.flush()
        return CurrentUser(user.id, user.username, user.display_name, {})


def login(request: Request, username: str, password: str) -> CurrentUser:
    username = (username or "").strip().lower()[:40]
    limiter.check(f"login:{_client_ip(request)}", 20, 900, message="Too many sign-in attempts. Try again in a few minutes.")
    limiter.check(f"login-user:{username}", 10, 900, message="Too many sign-in attempts for this account. Try again later.")
    with session_scope() as db:
        user = db.scalar(select(User).where(User.username == username))
        if user is None:
            verify_password(password or "", _DUMMY_HASH)
            raise HTTPException(401, "Wrong username or password.")
        if not verify_password(password or "", user.password_hash):
            raise HTTPException(401, "Wrong username or password.")
        return CurrentUser(user.id, user.username, user.display_name, dict(user.preferences or {}))


def change_password(user: CurrentUser, current: str, new: str) -> None:
    validate_credentials(user.username, new)
    with session_scope() as db:
        row = db.get(User, user.id)
        if row is None or not verify_password(current or "", row.password_hash):
            raise HTTPException(401, "Current password is wrong.")
        row.password_hash = hash_password(new)
        db.execute(delete(UserSession).where(UserSession.user_id == user.id))  # sign out everywhere


ALLOWED_PREFS = {"kb_mode": {"auto", "always", "off"}, "show_panel": {True, False}}


def update_preferences(user: CurrentUser, prefs: dict) -> dict:
    clean = {}
    for k, v in (prefs or {}).items():
        if k in ALLOWED_PREFS and v in ALLOWED_PREFS[k]:
            clean[k] = v
    with session_scope() as db:
        row = db.get(User, user.id)
        merged = {**(row.preferences or {}), **clean}
        row.preferences = merged
        return merged


def update_profile(user: CurrentUser, display_name: str) -> str:
    name = (display_name or "").strip()[:80]
    if not name:
        raise HTTPException(422, "Display name cannot be empty.")
    with session_scope() as db:
        db.get(User, user.id).display_name = name
    return name
