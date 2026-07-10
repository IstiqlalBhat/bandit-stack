import pytest


@pytest.fixture(autouse=True)
def mock_service_keys(monkeypatch):
    monkeypatch.setenv("PROXY_CLIENT_TEST_KEY", "client-test-key")
    monkeypatch.setenv("PROXY_ADMIN_TEST_KEY", "admin-test-key")
    monkeypatch.setenv("DECISION_API_TEST_KEY", "decision-test-key")
