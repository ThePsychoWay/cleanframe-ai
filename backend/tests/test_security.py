from app.auth.security import hash_password, password_needs_rehash, verify_password


def test_hash_round_trip_and_wrong_password() -> None:
    hashed = hash_password("a long enough passphrase")
    assert verify_password(hashed, "a long enough passphrase") is True
    assert verify_password(hashed, "a different passphrase") is False


def test_hashes_are_salted() -> None:
    assert hash_password("same-password-twice") != hash_password("same-password-twice")


def test_malformed_hash_fails_closed() -> None:
    assert verify_password("not-an-argon2-hash", "anything") is False
    assert verify_password("", "anything") is False


def test_fresh_hash_does_not_need_rehash() -> None:
    assert password_needs_rehash(hash_password("a long enough passphrase")) is False
