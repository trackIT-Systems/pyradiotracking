import logging
import os
import socket
import struct
import sys
import tempfile

import pytest

from radiotracking import journal
from radiotracking.journal import (
    JournalHandler,
    journal_field,
    log_handler,
    stderr_is_journal,
    syslog_priority,
)

unix_only = pytest.mark.skipif(sys.platform.startswith("win"), reason="no Unix sockets")


def parse_datagram(data: bytes) -> dict[str, str]:
    """The fields of one native-protocol datagram."""
    fields = {}
    while data:
        line, _, rest = data.partition(b"\n")
        if b"=" in line:
            key, _, value = line.partition(b"=")
            fields[key.decode()] = value.decode()
            data = rest
        else:
            (length,) = struct.unpack("<Q", rest[:8])
            fields[line.decode()] = rest[8 : 8 + length].decode()
            assert rest[8 + length : 9 + length] == b"\n"
            data = rest[9 + length :]
    return fields


@pytest.fixture
def journald():
    """A datagram socket standing in for journald; yields (path, receive)."""
    # AF_UNIX paths are limited to ~100 bytes, pytest's tmp_path can be longer
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        path = os.path.join(directory, "socket")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        server.bind(path)
        server.settimeout(1)

        def receive() -> dict[str, str]:
            return parse_datagram(server.recv(65536))

        yield path, receive
        server.close()


@pytest.fixture
def make_logger():
    """Attaches a handler to a test logger; the handler (its socket) is closed afterwards."""
    logger = logging.getLogger("test.journal")

    def make(handler: logging.Handler) -> logging.Logger:
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        logger.handlers[:] = [handler]
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        return logger

    yield make
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()


@pytest.mark.parametrize("level, priority", [(logging.CRITICAL, 2), (logging.ERROR, 3), (logging.WARNING, 4), (logging.INFO, 6), (logging.DEBUG, 7), (25, 6)])
def test_syslog_priority(level, priority):
    assert syslog_priority(level) == priority


def test_journal_field_plain_and_multiline():
    assert journal_field("MESSAGE", "hello") == b"MESSAGE=hello\n"
    assert journal_field("MESSAGE", "a\nb") == b"MESSAGE\n" + struct.pack("<Q", 3) + b"a\nb\n"


@unix_only
def test_handler_sends_fields(journald, make_logger):
    path, receive = journald
    logger = make_logger(JournalHandler("radiotracking", path))

    logger.warning("Cannot open %s", "/dev/ttyUSB0")
    fields = receive()

    assert fields["MESSAGE"] == "test.journal: Cannot open /dev/ttyUSB0"
    assert fields["PRIORITY"] == "4"
    assert fields["SYSLOG_IDENTIFIER"] == "radiotracking"
    assert fields["CODE_FILE"] == __file__
    assert fields["CODE_FUNC"] == "test_handler_sends_fields"
    assert int(fields["CODE_LINE"]) > 0


@unix_only
def test_traceback_is_one_entry(journald, make_logger):
    path, receive = journald
    logger = make_logger(JournalHandler("radiotracking", path))

    try:
        raise ValueError("broken")
    except ValueError:
        logger.exception("Failed")
    fields = receive()

    assert fields["PRIORITY"] == "3"
    assert fields["MESSAGE"].startswith("test.journal: Failed\nTraceback")
    assert fields["MESSAGE"].endswith("ValueError: broken")


@unix_only
def test_send_failure_falls_back_to_stderr(journald, make_logger, capsys):
    path, _ = journald
    handler = JournalHandler("radiotracking", path)
    logger = make_logger(handler)
    handler.socket.close()

    logger.error("Lost")

    assert capsys.readouterr().err == "<3>test.journal: Lost\n"


@unix_only
def test_missing_socket_raises(tmp_path):
    with pytest.raises(OSError):
        JournalHandler("radiotracking", str(tmp_path / "missing"))


def test_stderr_is_journal(monkeypatch, tmp_path):
    monkeypatch.delenv("JOURNAL_STREAM", raising=False)
    assert not stderr_is_journal()

    with open(tmp_path / "stderr", "w") as stderr:
        stat = os.fstat(stderr.fileno())
        monkeypatch.setattr(journal.sys, "stderr", stderr)
        monkeypatch.setenv("JOURNAL_STREAM", f"{stat.st_dev}:{stat.st_ino}")
        assert stderr_is_journal()
        monkeypatch.setenv("JOURNAL_STREAM", f"{stat.st_dev}:{stat.st_ino + 1}")
        assert not stderr_is_journal()
        monkeypatch.setenv("JOURNAL_STREAM", "garbage")
        assert not stderr_is_journal()


@unix_only
def test_log_handler_uses_journal_under_systemd(monkeypatch, journald, make_logger):
    path, receive = journald
    monkeypatch.setattr(journal, "stderr_is_journal", lambda: True)
    monkeypatch.setattr(JournalHandler.__init__, "__defaults__", (path,))

    handler = log_handler("radiotracking")
    assert isinstance(handler, JournalHandler)
    make_logger(handler).info("Started")
    fields = receive()

    assert fields["MESSAGE"] == "test.journal: Started"
    assert fields["SYSLOG_IDENTIFIER"] == "radiotracking"
    assert fields["PRIORITY"] == "6"


def test_log_handler_on_terminal(monkeypatch):
    monkeypatch.setattr(journal, "stderr_is_journal", lambda: False)

    handler = log_handler("radiotracking")

    assert type(handler) is logging.StreamHandler
    assert handler.format(logging.makeLogRecord({"name": "radiotracking.x", "levelname": "WARNING", "msg": "Careful"})) == "WARNING radiotracking.x: Careful"


@unix_only
def test_log_handler_without_journald(monkeypatch, tmp_path):
    monkeypatch.setattr(journal, "stderr_is_journal", lambda: True)
    monkeypatch.setattr(JournalHandler.__init__, "__defaults__", (str(tmp_path / "missing"),))

    handler = log_handler("radiotracking")

    assert type(handler) is logging.StreamHandler


@pytest.mark.parametrize("verbose, level", [(0, logging.WARNING), (1, logging.INFO), (2, logging.DEBUG), (3, logging.DEBUG)])
def test_setup_logging_replaces_the_root_handlers(monkeypatch, verbose, level):
    monkeypatch.setattr(journal, "stderr_is_journal", lambda: False)
    root = logging.getLogger()
    handlers, root_level = root.handlers[:], root.level
    try:
        journal.setup_logging(verbose)
        journal.setup_logging(verbose)

        assert len(root.handlers) == 1
        assert type(root.handlers[0]) is logging.StreamHandler
        assert root.level == level
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers[:] = handlers
        root.setLevel(root_level)
