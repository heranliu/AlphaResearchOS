"""Tests may exercise loopback HTTP, but must never reach a live provider."""

import ipaddress
import socket
import urllib.parse

import pytest


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    from alpharesearchos import http_transport, jev_client, provider_transport

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

    # HTTP workers run in fresh interpreters and cannot inherit socket patches.
    # Check the destination in the parent before any subprocess can be started.
    read = http_transport.bounded_read

    def checked_read(request, **kwargs):
        local(urllib.parse.urlsplit(request.full_url).hostname)
        return read(request, **kwargs)

    for module in (http_transport, provider_transport, jev_client):
        monkeypatch.setattr(module, "bounded_read", checked_read)
    # Local fixtures must not accidentally route through an inherited proxy.
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,::1")
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1,::1")
