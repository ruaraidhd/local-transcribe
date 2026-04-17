"""Tests for _RangeHTTPRequestHandler._parse_range and the audio HTTP server."""
from __future__ import annotations

import os
import socketserver
import tempfile
import threading
import urllib.request

import pytest

from gui import _RangeHTTPRequestHandler, _make_audio_handler


# ---------------------------------------------------------------------------
# Unit tests for _parse_range
# ---------------------------------------------------------------------------

def parse(header, size):
    return _RangeHTTPRequestHandler._parse_range(header, size)


SIZE = 1000


@pytest.mark.parametrize("header,size,expected", [
    ("bytes=0-99",    SIZE, (0, 99)),
    ("bytes=0-",      SIZE, (0, SIZE - 1)),
    ("bytes=-500",    SIZE, (SIZE - 500, SIZE - 1)),
    ("bytes=100-200", SIZE, (100, 200)),
    ("bytes=100-999999", 500, (100, 499)),   # end clamped to size-1
    ("invalid",       SIZE, (None, None)),
    ("bytes=abc-def", SIZE, (None, None)),
    ("chunks=0-99",   SIZE, (None, None)),   # wrong unit
])
def test_parse_range(header, size, expected):
    assert parse(header, size) == expected


def test_parse_range_suffix_range_clipped():
    """bytes=-500 with size=300 clamps start to 0 when suffix > size.

    RFC 9110 says: if the suffix-length exceeds the file size, the range
    covers the whole file (start=0).
    """
    start, end = parse("bytes=-500", 300)
    assert start == 0
    assert end == 299


# ---------------------------------------------------------------------------
# Integration test: spin up the server and make real HTTP requests
# ---------------------------------------------------------------------------

def _find_free_port():
    with socketserver.TCPServer(("127.0.0.1", 0), None) as s:
        return s.server_address[1]


def test_server_full_file():
    """GET without Range header returns full file with 200."""
    content = b"Hello Verbatim audio bytes"
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(content)
        fpath = f.name

    try:
        file_map = {"test-token": fpath}
        handler = _make_audio_handler(file_map)
        server = socketserver.TCPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/audio/test-token")
            assert resp.status == 200
            assert resp.read() == content
        finally:
            server.shutdown()
    finally:
        os.unlink(fpath)


def test_server_range_request():
    """GET with Range header returns partial content with 206."""
    content = b"ABCDEFGHIJ"  # 10 bytes, 0-indexed
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(content)
        fpath = f.name

    try:
        file_map = {"range-token": fpath}
        handler = _make_audio_handler(file_map)
        server = socketserver.TCPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()

        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/audio/range-token",
                headers={"Range": "bytes=2-5"},
            )
            resp = urllib.request.urlopen(req)
            assert resp.status == 206
            body = resp.read()
            assert body == b"CDEF"  # bytes 2,3,4,5
        finally:
            server.shutdown()
    finally:
        os.unlink(fpath)


def test_server_unknown_token_returns_404():
    """Requests for unknown tokens should return 404."""
    file_map: dict = {}
    handler = _make_audio_handler(file_map)
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/audio/no-such-token")
        assert exc_info.value.code == 404
    finally:
        server.shutdown()
