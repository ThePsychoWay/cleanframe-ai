"""Authentication dependencies for protected routes."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.auth.security import decode_access_token
from app.config import get_settings
from app.db import get_db
from app.logging_config import bind_context
from app.models import User

NOT_AUTHENTICATED = "You need to log in to do that."
SESSION_INVALID = "Your session is invalid or has expired. Please log in again."


def get_current_user(request: Request, db: Annotated[Session, Depends(get_db)]) -> User:
    token = request.cookies.get(get_settings().session_cookie_name)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, NOT_AUTHENTICATED)

    user_id = decode_access_token(token)
    user = db.get(User, user_id) if user_id else None
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, SESSION_INVALID)
    bind_context(user_id=user.id)
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_admin(user: CurrentUser) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You don't have access to this page.")
    return user
