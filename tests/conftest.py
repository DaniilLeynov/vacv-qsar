import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def guard(*args, **kwargs):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
