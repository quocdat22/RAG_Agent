import pytest

from rag import retry_utils
from rag.retry_utils import retry_with_backoff, is_retryable_error


def test_retry_with_backoff_retries_until_success(monkeypatch):
    delays: list[float] = []

    # Avoid real sleeping during tests
    monkeypatch.setattr(retry_utils.time, "sleep", lambda d: delays.append(d))

    attempts = {"count": 0}

    @retry_with_backoff(
        max_retries=2,
        initial_delay=0.1,
        exponential_base=2.0,
        retryable_exceptions=(RuntimeError,),
    )
    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("connection lost")
        return "ok"

    assert flaky() == "ok"
    assert attempts["count"] == 3  # initial try + 2 retries
    assert delays == [0.1, 0.2]  # exponential backoff applied


def test_retry_with_backoff_non_retryable_error_no_retry(monkeypatch):
    delays: list[float] = []
    monkeypatch.setattr(retry_utils.time, "sleep", lambda d: delays.append(d))

    attempts = {"count": 0}

    @retry_with_backoff(
        max_retries=3,
        retryable_exceptions=(ValueError,),
        check_retryable=True,
    )
    def fails_fast():
        attempts["count"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        fails_fast()

    assert attempts["count"] == 1  # should not retry non-retryable errors
    assert delays == []  # no backoff when not retrying


def test_is_retryable_error_detects_network_and_status_code():
    assert is_retryable_error(ConnectionError("timeout")) is True

    class ResponseError(Exception):
        def __init__(self, status_code):
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    assert is_retryable_error(ResponseError(503)) is True
    assert is_retryable_error(ResponseError(400)) is False

