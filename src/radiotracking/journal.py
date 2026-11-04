"""Logging to the systemd journal, following tsOS's logging convention (tsOS-base docs/logging.md).

Under systemd every record goes to the journal through python3-systemd's JournalHandler, with its
priority, the program's SYSLOG_IDENTIFIER and the source location as separate fields; the message
itself is only ``<logger name>: <message>``. On a terminal, in tests, or without python3-systemd (a C
extension from Debian that does not install on macOS) records go to stderr as
``<LEVEL> <logger name>: <message>``.
"""

from __future__ import annotations

import logging
import os
import sys

try:
    from systemd.journal import JournalHandler
except ImportError:
    JournalHandler = None


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


def log_handler(identifier: str | None = None) -> logging.Handler:
    """The journal handler under systemd, a stderr handler otherwise.

    ``identifier`` is the SYSLOG_IDENTIFIER, the program's name (default: the executable's name).
    """
    if JournalHandler is not None and stderr_is_journal():
        identifier = identifier or os.path.basename(sys.argv[0]) or "python3"
        handler: logging.Handler = JournalHandler(SYSLOG_IDENTIFIER=identifier)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        return handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    return handler


def setup_logging(verbose: int = 0) -> None:
    """Logs at WARNING, INFO with one ``-v``, DEBUG with two, to the journal or stderr (see ``log_handler``).

    Called by every process: an analyzer process replaces the handler it inherited.
    """
    level = max(logging.DEBUG, logging.WARNING - verbose * 10)
    logging.basicConfig(level=level, handlers=[log_handler("radiotracking")], force=True)
