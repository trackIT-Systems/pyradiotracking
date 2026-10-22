"""Logging to the systemd journal, following tsOS's logging convention (tsOS-base docs/logging.md).

Under systemd every record goes to the journal over its native protocol, with its priority, the
program's SYSLOG_IDENTIFIER and the source location as separate fields; the message itself is only
``<logger name>: <message>``. On a terminal (or in tests) records go to stderr as
``<LEVEL> <logger name>: <message>``.
"""

from __future__ import annotations

import logging
import os
import socket
import struct
import sys

JOURNAL_SOCKET = "/run/systemd/journal/socket"

# Python level -> syslog priority (RFC 5424); 0, 1 and 5 are not used
PRIORITIES = ((logging.CRITICAL, 2), (logging.ERROR, 3), (logging.WARNING, 4), (logging.INFO, 6))
DEBUG_PRIORITY = 7

# Longer messages are cut, so a record always fits in one datagram
MAX_MESSAGE = 32 * 1024


def syslog_priority(levelno: int) -> int:
    for level, priority in PRIORITIES:
        if levelno >= level:
            return priority
    return DEBUG_PRIORITY


def stderr_is_journal() -> bool:
    """Whether systemd connected stderr to the journal.

    systemd sets JOURNAL_STREAM to the device and inode of that stream; a process whose stderr was
    redirected since (or a child that inherited the variable) has a different stderr. See systemd.exec(5).
    """
    stream = os.environ.get("JOURNAL_STREAM")
    if not stream:
        return False
    try:
        dev, ino = (int(part) for part in stream.split(":"))
        stat = os.fstat(sys.stderr.fileno())
    except (ValueError, OSError, AttributeError):
        return False
    return (stat.st_dev, stat.st_ino) == (dev, ino)


def journal_field(key: str, value: str) -> bytes:
    """One field of the native journal protocol; a value with a newline is sent with its length."""
    data = value.encode("utf-8", "replace")
    if b"\n" in data:
        return key.encode() + b"\n" + struct.pack("<Q", len(data)) + data + b"\n"
    return key.encode() + b"=" + data + b"\n"


class JournalHandler(logging.Handler):
    """Sends records to journald over its native protocol (https://systemd.io/JOURNAL_NATIVE_PROTOCOL/).

    A record that cannot be sent (journald restarting) is written to stderr instead, with its
    priority as ``<N>`` prefix, which journald also understands.
    """

    def __init__(self, identifier: str, path: str = JOURNAL_SOCKET) -> None:
        super().__init__()
        self.identifier = identifier
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            self.socket.connect(path)
        except OSError:
            self.socket.close()
            raise

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)[:MAX_MESSAGE]
            priority = str(syslog_priority(record.levelno))
            fields = (
                ("MESSAGE", message),
                ("PRIORITY", priority),
                ("SYSLOG_IDENTIFIER", self.identifier),
                ("CODE_FILE", record.pathname),
                ("CODE_LINE", str(record.lineno)),
                ("CODE_FUNC", record.funcName),
            )
            try:
                self.socket.send(b"".join(journal_field(key, value) for key, value in fields))
            except OSError:
                sys.stderr.write(f"<{priority}>{message}\n")
                sys.stderr.flush()
        except Exception:  # noqa: BLE001 - logging must not raise into the caller
            self.handleError(record)

    def close(self) -> None:
        self.socket.close()
        super().close()


def log_handler(identifier: str | None = None) -> logging.Handler:
    """The journal handler under systemd, a stderr handler otherwise (or when journald is unreachable).

    ``identifier`` is the SYSLOG_IDENTIFIER, the program's name (default: the executable's name).
    """
    if stderr_is_journal():
        try:
            handler: logging.Handler = JournalHandler(identifier or os.path.basename(sys.argv[0]) or "python3")
            handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
            return handler
        except OSError:
            pass
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    return handler


def setup_logging(verbose: int = 0) -> None:
    """Logs at WARNING, INFO with one ``-v``, DEBUG with two, to the journal or stderr (see ``log_handler``).

    Called by every process: an analyzer process replaces the handler it inherited, and gets its own socket.
    """
    level = max(logging.DEBUG, logging.WARNING - verbose * 10)
    logging.basicConfig(level=level, handlers=[log_handler("radiotracking")], force=True)
