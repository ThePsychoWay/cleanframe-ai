"""Registration, login, logout and the current-user endpoint.

The session lives in an httpOnly cookie so page scripts never see the token.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth.deps import CurrentUser
from app.auth.security import (
    burn_password_check,
    create_access_token,
    hash_password,
    password_needs_rehash,
    verify_password,
)
from app.config import get_settings
from app.db import get_db
from app.logging_config import bind_context
from app.models import User
from app.ratelimit import Limiter, client_ip, email_key
from app.schemas.auth import LoginRequest, RegisterRequest, UserResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])

INVALID_CREDENTIALS = "Incorrect email or password."
EMAIL_TAKEN = "An account with this email already exists. Log in instead."

Db = Annotated[Session, Depends(get_db)]


def _set_session_cookie(response: Response, user: User) -> None:
    settings = get_settings()
    response.set_cookie(
        key=settings.session_cookie_name,
        value=create_access_token(user.id),
        max_age=settings.jwt_expire_minutes * 60,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, request: Request, response: Response, db: Db, limiter: Limiter) -> User:
    limiter.enforce("register_ip", client_ip(request))
    if db.scalar(select(User.id).where(User.email == payload.email)) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, EMAIL_TAKEN)

    user = User(email=payload.email, name=payload.name, password_hash=hash_password(payload.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        # Two sign-ups with the same email raced; the unique index decided.
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, EMAIL_TAKEN) from None

    bind_context(user_id=user.id)
    logger.info("user registered")
    _set_session_cookie(response, user)
    return user


@router.post("/login", response_model=UserResponse)
def login(payload: LoginRequest, request: Request, response: Response, db: Db, limiter: Limiter) -> User:
    # Two limits: attempts from one address, and failures against one account (whoever sends them). Both count
    # unknown emails too, so being locked out says nothing about whether an account exists.
    limiter.enforce("login_ip", client_ip(request))
    account = email_key(payload.email)
    limiter.require_clear("login_account", account)

    user = db.scalar(select(User).where(User.email == payload.email))

    if user is None:
        burn_password_check(payload.password)
        limiter.record("login_account", account)
        logger.info("login failed")  # no email in the log: it would say which addresses people try
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, INVALID_CREDENTIALS)
    if not verify_password(user.password_hash, payload.password) or not user.is_active:
        limiter.record("login_account", account)
        logger.info("login failed", extra={"user_id": str(user.id)})
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, INVALID_CREDENTIALS)

    limiter.clear("login_account", account)
    if password_needs_rehash(user.password_hash):
        user.password_hash = hash_password(payload.password)
        db.commit()

    bind_context(user_id=user.id)
    logger.info("login succeeded")
    _set_session_cookie(response, user)
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout() -> Response:
    settings = get_settings()
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.delete_cookie(
        key=settings.session_cookie_name,
        path="/",
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )
    return response


@router.get("/me", response_model=UserResponse)
def me(user: CurrentUser) -> User:
    return user
