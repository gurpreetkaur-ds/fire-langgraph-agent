"""Accounts API: sign-up / sign-in / sign-out, the current user, preferences and public configuration."""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import auth
from .auth import CurrentUser, current_user, require_csrf, write_user
from .settings import get_app_settings

router = APIRouter(prefix="/api", tags=["accounts"])
NO_STORE = {"Cache-Control": "no-store"}


class SignupIn(BaseModel):
    username: str = Field(max_length=40)
    password: str = Field(max_length=auth.MAX_PASSWORD)
    display_name: str = Field(default="", max_length=80)
    invite_code: Optional[str] = Field(default=None, max_length=200)


class LoginIn(BaseModel):
    username: str = Field(max_length=40)
    password: str = Field(max_length=auth.MAX_PASSWORD)


class PrefsIn(BaseModel):
    kb_mode: Optional[Literal["auto", "always", "off"]] = None
    show_panel: Optional[bool] = None


class ProfileIn(BaseModel):
    display_name: Optional[str] = Field(default=None, max_length=80)
    preferences: Optional[PrefsIn] = None


class PasswordIn(BaseModel):
    current_password: str = Field(max_length=auth.MAX_PASSWORD)
    new_password: str = Field(max_length=auth.MAX_PASSWORD)


@router.get("/config")
def public_config():
    s = get_app_settings()
    return JSONResponse({"signup_open": s.allow_signup, "invite_required": auth.invite_required(), "rag": s.public()}, headers=NO_STORE)


@router.post("/auth/signup", dependencies=[Depends(require_csrf)])
def signup(body: SignupIn, request: Request, response: Response):
    user = auth.signup(request, body.username, body.password, body.display_name, body.invite_code)
    auth.start_session(response, user.id)
    response.headers["Cache-Control"] = "no-store"
    return {"user": user.public()}


@router.post("/auth/login", dependencies=[Depends(require_csrf)])
def login(body: LoginIn, request: Request, response: Response):
    user = auth.login(request, body.username, body.password)
    auth.start_session(response, user.id)
    response.headers["Cache-Control"] = "no-store"
    return {"user": user.public()}


@router.post("/auth/logout", dependencies=[Depends(require_csrf)])
def logout(request: Request, response: Response):
    auth.end_session(request, response)
    return {"ok": True}


@router.get("/me")
def me(user: CurrentUser = Depends(current_user)):
    return JSONResponse({"user": user.public()}, headers=NO_STORE)


@router.patch("/me")
def update_me(body: ProfileIn, user: CurrentUser = Depends(write_user)):
    out = user.public()
    if body.display_name is not None:
        out["display_name"] = auth.update_profile(user, body.display_name)
    if body.preferences is not None:
        out["preferences"] = auth.update_preferences(user, body.preferences.model_dump(exclude_none=True))
    return {"user": out}


@router.post("/me/password")
def change_password(body: PasswordIn, request: Request, response: Response, user: CurrentUser = Depends(write_user)):
    auth.change_password(user, body.current_password, body.new_password)
    auth.start_session(response, user.id)  # other sessions were revoked; keep this browser signed in
    return {"ok": True}
