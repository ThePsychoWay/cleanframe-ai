from collections.abc import Iterator

from fastapi.testclient import TestClient


def _chunks(total: int, size: int = 64 * 1024) -> Iterator[bytes]:
    sent = 0
    while sent < total:
        yield b" " * min(size, total - sent)
        sent += size


def test_declared_oversize_is_refused(client: TestClient) -> None:
    response = client.post("/api/auth/login", content=b" " * (3 * 1024 * 1024), headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    assert response.json() == {"detail": "This request is too large."}


def test_streamed_oversize_is_refused_even_without_a_content_length(client: TestClient) -> None:
    # A generator body is sent chunked, so there's no Content-Length to check up front.
    response = client.post("/api/auth/login", content=_chunks(3 * 1024 * 1024), headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_a_garbage_content_length_is_a_bad_request(client: TestClient) -> None:
    response = client.post("/api/auth/login", content=b"{}", headers={"Content-Type": "application/json", "Content-Length": "abc"})
    assert response.status_code in {400, 422}  # rejected by the client library or by us; never processed


def test_normal_requests_pass_through(client: TestClient) -> None:
    response = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "wrong password here"})
    assert response.status_code == 401


def test_the_limit_response_still_carries_cors_headers(client: TestClient) -> None:
    response = client.post(
        "/api/auth/login",
        content=b" " * (3 * 1024 * 1024),
        headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"},
    )
    assert response.status_code == 413
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
