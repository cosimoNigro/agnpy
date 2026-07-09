import platform
import sys
from tempfile import TemporaryDirectory

import pytest
import zmq


@pytest.mark.parametrize(
    "capability",
    [
        "curve",
        "ipc",
    ],
)
def test_has_capability(capability):
    assert zmq.has(capability) is True


@pytest.mark.parametrize(
    "capability",
    [
        "draft",
    ],
)
def test_not_has_capability(capability):
    assert zmq.has(capability) is False


def test_tcp():
    with zmq.Context() as ctx:
        with ctx.socket(zmq.PUSH) as push, ctx.socket(zmq.PULL) as pull:
            iface = "tcp://127.0.0.1"
            port = push.bind_to_random_port(iface)
            url = f"{iface}:{port}"
            pull.connect(url)
            push.linger = pull.linger = 1_000
            push.sndtimeo = pull.rcvtimeo = 1_000
            push.send(b"msg")
            msg = pull.recv()
            assert msg == b"msg"


def test_ipc():
    with TemporaryDirectory() as td, zmq.Context() as ctx:
        with ctx.socket(zmq.PUSH) as push, ctx.socket(zmq.PULL) as pull:
            from pathlib import Path

            print(td, type(td), Path(td).exists())
            url = f"ipc://{td}/test_socket"
            push.bind(url)
            pull.connect(url)
            push.linger = pull.linger = 1_000
            push.sndtimeo = pull.rcvtimeo = 1_000
            push.send(b"msg")
            msg = pull.recv()
            assert msg == b"msg"


def test_backend():
    if platform.python_implementation() == "CPython":
        assert "zmq.backend.cython" in sys.modules
    else:
        assert "zmq.backend.cffi" in sys.modules
