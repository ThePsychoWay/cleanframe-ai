import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, StringConstraints, field_validator

PASSWORD_MIN_LENGTH = 10
PASSWORD_MAX_LENGTH = 128  # bounds hashing cost; long passphrases are still fine

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class _EmailMixin(BaseModel):
    email: EmailStr

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        return value.strip().lower()


class RegisterRequest(_EmailMixin):
    name: Name
    password: Annotated[str, StringConstraints(min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH)]


class LoginRequest(_EmailMixin):
    password: Annotated[str, StringConstraints(min_length=1, max_length=PASSWORD_MAX_LENGTH)]


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    name: str
    role: str
    created_at: datetime
