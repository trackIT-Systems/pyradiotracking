#!/usr/bin/env python3
"""
Exercise the SDR watchdog shutdown path in radiotracking.__main__.Runner.check_analyzers.

Historical failure mode: after "SDR … timed out.", the parent called terminate() then
join() with no timeout. A child that ignored SIGTERM (like a wedged librtlsdr read)
could block join() forever.

This script replaces SignalAnalyzer with stubs for every configured device. The device
identified by --hung-device (default: 0) uses HungSignalAnalyzer: it ignores SIGTERM
and keeps a stale last_data_ts so the watchdog fires. All other devices use
HealthyStubAnalyzer: they refresh last_data_ts periodically so they stay "alive".

With the fix in Runner, the parent waits on join() (SIGTERM), then escalates to
SIGKILL for the hung device so the run should complete after the join timeout unless
you disable the unblock helper and rely entirely on Runner.

Optional --unblock-after-s (>0): a background thread SIGKILLs the hung stub sooner.

Usage (from repository root, with dependencies installed):

  PYTHONPATH=src python3 scripts/repro_sdr_watchdog_join_hang.py

Multiple devices (include the hung device id, default ``0``, so one stub simulates the hang):

  PYTHONPATH=src python3 scripts/repro_sdr_watchdog_join_hang.py -d 0 1 2 -c 0 0 0

Unknown flags (e.g. ``-d``, ``-c``, ``--config``) are forwarded to radiotracking's parser.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# Repository root (parent of scripts/)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _device_id(device: str | int) -> str:
    return str(device).strip()


def _is_hung_device(device: str | int, hung_device: str) -> bool:
    return _device_id(device) == _device_id(hung_device)


def _detach_from_parent_signal_handlers() -> None:
    """
    Forked/spawned children inherit the parent's SIGTERM/SIGINT handlers (Runner wires
    them to terminate()). If a child receives SIGTERM while that handler is still
    active, terminate() runs in the child and tries Process.join() on siblings, which
    raises AssertionError: can only join a child process.
    """
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


class HungSignalAnalyzer(multiprocessing.Process):
    """
    Stand-in for a wedged SDR worker: ignores SIGTERM and never refreshes last_data_ts
    (parent uses a stale shared timestamp).
    """

    def __init__(self, signal_queue, last_data_ts, **kwargs):
        super().__init__()
        self.signal_queue = signal_queue
        self.last_data_ts = last_data_ts
        self.device = kwargs["device"]
        self.calibration_db = kwargs["calibration_db"]
        self.sdr_timeout_s = kwargs["sdr_timeout_s"]
        self.sdr_max_restart = kwargs["sdr_max_restart"]
        try:
            self.device_index = int(self.device)
        except ValueError:
            self.device_index = 0

    def run(self):
        _detach_from_parent_signal_handlers()
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            time.sleep(3600)


class HealthyStubAnalyzer(multiprocessing.Process):
    """
    Stand-in for a responsive SDR: keeps last_data_ts current so the watchdog does not
    fire for this device.
    """

    def __init__(self, signal_queue, last_data_ts, **kwargs):
        super().__init__()
        self.signal_queue = signal_queue
        self.last_data_ts = last_data_ts
        self.device = kwargs["device"]
        self.calibration_db = kwargs["calibration_db"]
        self.sdr_timeout_s = kwargs["sdr_timeout_s"]
        self.sdr_max_restart = kwargs["sdr_max_restart"]
        try:
            self.device_index = int(self.device)
        except ValueError:
            self.device_index = 0

    def run(self):
        _detach_from_parent_signal_handlers()
        while True:
            self.last_data_ts.value = datetime.datetime.now().timestamp()
            time.sleep(0.25)


def parse_script_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--unblock-after-s",
        type=float,
        default=8.0,
        help="Send SIGKILL to the hung stub after this many seconds (0 = disable).",
    )
    p.add_argument(
        "--hung-device",
        default="0",
        help="Device id (same as -d / --device) whose stub ignores SIGTERM and times out (default: 0).",
    )
    return p.parse_known_args()


def main() -> None:
    script_args, rt_extra = parse_script_args()
    hung_device = script_args.hung_device

    rt_argv = [
        "repro_sdr_watchdog_join_hang",
        "--config",
        str(_REPO_ROOT / "etc" / "radiotracking.ini"),
        "--sdr-max-restart",
        "0",
        "--sdr-timeout-s",
        "1",
    ]
    rt_argv.extend(rt_extra)
    sys.argv = rt_argv

    from radiotracking.__main__ import Runner

    class ReproRunner(Runner):
        def create_and_start(self, device, calibration_db, sdr_max_restart=None):
            import argparse as ap

            dargs = ap.Namespace(**vars(self.args))
            dargs.device = device
            dargs.calibration_db = calibration_db
            if sdr_max_restart is not None:
                dargs.sdr_max_restart = sdr_max_restart

            if _is_hung_device(device, hung_device):
                last_data_ts = multiprocessing.Value("d", datetime.datetime.now().timestamp() - 86400.0)
                analyzer = HungSignalAnalyzer(signal_queue=self.connector.q, last_data_ts=last_data_ts, **vars(dargs))
            else:
                last_data_ts = multiprocessing.Value("d", 0.0)
                analyzer = HealthyStubAnalyzer(signal_queue=self.connector.q, last_data_ts=last_data_ts, **vars(dargs))

            analyzer.start()

            if not self.args.sdr_dynamic_scheduling:
                try:
                    cpu_core = analyzer.device_index % multiprocessing.cpu_count()
                    out = subprocess.check_output(["taskset", "-p", "-c", str(cpu_core), str(analyzer.pid)])
                    for line in out.decode().splitlines():
                        logging.getLogger("radiotracking").info("SDR %s CPU affinity: %s", analyzer.device, line)
                except FileNotFoundError:
                    logging.getLogger("radiotracking").warning(
                        "SDR %s CPU affinity: failed to configure (taskset missing?)",
                        analyzer.device,
                    )

            return analyzer

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("repro")

    runner = ReproRunner()
    runner.start_analyzers()
    if not runner.analyzers:
        log.error("No analyzers started; check --device / calibration counts.")
        sys.exit(1)

    hung = next((a for a in runner.analyzers if isinstance(a, HungSignalAnalyzer)), None)
    if hung is None:
        log.warning(
            "No hung stub (HungSignalAnalyzer) among analyzers — device %r is not in %s. "
            "Watchdog will not fire unless you include that device id (e.g. -d 0 1 with default --hung-device 0).",
            hung_device,
            [getattr(a, "device", "?") for a in runner.analyzers],
        )

    unblock_after = script_args.unblock_after_s
    if unblock_after > 0 and hung is not None:

        def unblock():
            time.sleep(unblock_after)
            if hung.is_alive():
                log.warning(
                    "Unblock: sending SIGKILL to hung stub SDR %s pid=%s (SIGTERM was ignored).",
                    hung.device,
                    hung.pid,
                )
                try:
                    os.kill(hung.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        threading.Thread(target=unblock, name="repro-unblock", daemon=True).start()
        log.info(
            "Started %s stub analyzer(s); hung device=%r pid=%s (others are healthy stubs). "
            "Calling check_analyzers(); unblock SIGKILL in %.1fs if join stalls.",
            len(runner.analyzers),
            hung.device,
            hung.pid,
            unblock_after,
        )
    elif hung is not None:
        log.info(
            "Started %s stub analyzer(s); hung device=%r pid=%s. Calling check_analyzers() "
            "(no early SIGKILL thread; Runner escalates after join timeout).",
            len(runner.analyzers),
            hung.device,
            hung.pid,
        )
    else:
        log.info(
            "Started %s stub analyzer(s). Calling check_analyzers().",
            len(runner.analyzers),
        )

    runner.check_analyzers()
    log.info("check_analyzers returned.")


if __name__ == "__main__":
    main()
