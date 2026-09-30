"""Tests may exercise loopback HTTP, but must never reach a live provider."""

import ipaddress
import socket

import pytest


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    connect, connect_ex, resolve = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo

    def local(host):
        if isinstance(host, bytes):
            host = host.decode("ascii")
        if host in {None, "localhost"}:
            return
        try:
            if ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            pass
        raise AssertionError("Tests must mock external network calls; only loopback is allowed")

    def checked_connect(sock, address):
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            local(address[0])
        return connect(sock, address)

    def checked_connect_ex(sock, address):
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            local(address[0])
        return connect_ex(sock, address)

    def checked_resolve(host, *args, **kwargs):
        local(host)
        return resolve(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", checked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", checked_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", checked_resolve)
