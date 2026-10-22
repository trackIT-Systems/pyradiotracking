import datetime
import signal
import sys
import types

import pytest


def _install_rtlsdr_stub():
    """Provide a stand-in for pyrtlsdr on machines without librtlsdr (the import loads the shared library)."""
    try:
        import rtlsdr  # noqa: F401

        return
    except ImportError:
        pass

    class LibUSBError(Exception):
        pass

    class RtlSdr:
        @staticmethod
        def get_device_index_by_serial(serial):
            raise LibUSBError(f"no device with serial {serial}")

    pkg = types.ModuleType("rtlsdr")
    mod = types.ModuleType("rtlsdr.rtlsdr")
    mod.RtlSdr = RtlSdr
    mod.LibUSBError = LibUSBError
    lib = types.ModuleType("rtlsdr.librtlsdr")
    pkg.rtlsdr = mod
    pkg.librtlsdr = lib
    sys.modules.update({"rtlsdr": pkg, "rtlsdr.rtlsdr": mod, "rtlsdr.librtlsdr": lib})


_install_rtlsdr_stub()

from radiotracking import Signal  # noqa: E402

T0 = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)


@pytest.fixture(autouse=True)
def keep_test_logging(monkeypatch):
    """The runner and the analyzers replace the root handlers, which would remove the handler of caplog."""
    import radiotracking.__main__
    import radiotracking.analyze

    monkeypatch.setattr(radiotracking.__main__, "setup_logging", lambda verbose=0: None)
    monkeypatch.setattr(radiotracking.analyze, "setup_logging", lambda verbose=0: None)


@pytest.fixture
def make_signal():
    """Factory for Signals with sensible defaults."""

    def factory(device="0", ts=T0, frequency=150_100_000.0, duration=0.02, max_dbw=-50.0, avg_dbw=-55.0, **kw):
        return Signal(
            device,
            ts,
            frequency,
            duration,
            max_dbw,
            avg_dbw,
            kw.get("std_db", 1.0),
            kw.get("noise_dbw", -80.0),
            kw.get("snr_db", 25.0),
        )

    return factory


@pytest.fixture
def restore_signal_handlers():
    """Runner() installs SIGINT/SIGTERM handlers, put the previous ones back."""
    old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    yield
    for s, h in old.items():
        signal.signal(s, h)
