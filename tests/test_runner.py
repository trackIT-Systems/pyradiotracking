import datetime
import json
import logging
import queue
import signal
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import radiotracking.__main__ as rt_main
from radiotracking import StateMessage
from radiotracking.__main__ import Runner


@pytest.fixture(autouse=True)
def clean_state(restore_signal_handlers):
    yield


@pytest.fixture
def make_runner(tmp_path, monkeypatch):
    def factory(*argv):
        ini = tmp_path / "none.ini"
        monkeypatch.setattr(sys, "argv", ["radiotracking", "--config", str(ini), "--path", str(tmp_path / "out"), *argv])
        return Runner()

    return factory


class FakeValue:
    def __init__(self, value=0.0):
        self.value = value


class FakeAnalyzer:
    def __init__(self, device="0", alive=True, streaming_ts=0.0, last_data_ts=0.0, started_at=None, max_restart=3, timeout=2):
        self.device = device
        self.calibration_db = 0.0
        self.sdr_max_restart = max_restart
        self.sdr_timeout_s = timeout
        self.streaming_ts = FakeValue(streaming_ts)
        self.last_data_ts = FakeValue(last_data_ts)
        self.started_at = datetime.datetime.now().timestamp() if started_at is None else started_at
        self.signal_queue = queue.Queue()
        self.alive = alive
        self.exitcode = None if alive else 0
        self.pid = 4711
        self.terminated = False
        self.killed = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False
        self.exitcode = -signal.SIGTERM

    def kill(self):
        self.killed = True
        self.alive = False
        self.exitcode = -signal.SIGKILL

    def join(self, timeout=None):
        pass


class TestInit:
    def test_defaults(self, make_runner):
        r = make_runner()
        assert r.running
        assert r.analyzers == []
        assert r.args.calibration == [0.0]

    def test_calibration_defaults_to_zero_per_device(self, make_runner):
        assert make_runner("-d", "0", "1", "2").args.calibration == [0.0, 0.0, 0.0]

    def test_calibration_must_match_devices(self, make_runner):
        with pytest.raises(SystemExit) as exc:
            make_runner("-d", "0", "1", "-c", "1.0")
        assert exc.value.code == 1

    def test_no_devices(self, make_runner):
        with pytest.raises(SystemExit) as exc:
            make_runner("-d")
        assert exc.value.code == 1

    @pytest.mark.parametrize("devices", [["0", "0"], ["0", "1", "0"]])
    def test_duplicate_devices(self, make_runner, caplog, devices):
        with pytest.raises(SystemExit) as exc:
            make_runner("-d", *devices)
        assert exc.value.code == 1
        assert "['0'] are configured more than once" in caplog.text

    def test_default_analysis_block_does_not_warn(self, make_runner, caplog):
        with caplog.at_level(logging.WARNING):
            make_runner()
        assert "analysis block" not in caplog.text

    def test_short_analysis_block_warns(self, make_runner, caplog):
        with caplog.at_level(logging.WARNING):
            r = make_runner("--analysis-block-samples", "30208")
        assert r.args.analysis_block_samples == 30208
        assert "The analysis block (30208 samples, 0.101 s) is shorter than 6 times signal_max_duration_ms" in caplog.text

    def test_long_signals_warn_with_the_default_block(self, make_runner, caplog):
        with caplog.at_level(logging.WARNING):
            make_runner("--signal-max-duration-ms", "100")
        assert "shorter than 6 times signal_max_duration_ms (100 ms" in caplog.text

    def test_block_is_judged_in_time_for_a_wide_band(self, make_runner, caplog):
        with caplog.at_level(logging.WARNING):
            make_runner("--sample-rate", "1200000", "--fft-nperseg", "1024")
        assert "analysis block" not in caplog.text

    def test_installs_termination_handlers(self, make_runner):
        make_runner()
        assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
        assert signal.getsignal(signal.SIGINT) is not signal.SIG_DFL

    def test_exports_configuration(self, make_runner, tmp_path):
        make_runner("--export-config", "--station", "teststation")
        exported = list((tmp_path / "out" / socket.gethostname() / "radiotracking").glob("teststation_*.ini"))
        assert len(exported) == 1
        text = exported[0].read_text()
        assert "center_freq = 150150000" in text
        assert "dashboard" not in text

    def test_exported_configuration_can_be_read_back(self, make_runner, tmp_path):
        r = make_runner("--export-config", "--station", "teststation", "-f", "151000000")
        exported = next((tmp_path / "out").rglob("teststation_*.ini"))
        assert Runner.parser.parse_args(["--config", str(exported)]).center_freq == 151_000_000
        assert r.args.center_freq == 151_000_000

    def test_matcher_is_registered_as_consumer(self, make_runner):
        r = make_runner()
        assert r.matcher in r.connector.consumers
        assert r.matcher.signal_queue is r.connector.q


class TestCheckAnalyzers:
    @pytest.fixture
    def runner(self, make_runner, monkeypatch):
        r = make_runner()
        r.created = []

        def create(device, calibration_db, sdr_max_restart=None):
            a = FakeAnalyzer(device, max_restart=sdr_max_restart if sdr_max_restart is not None else 3)
            r.created.append(a)
            return a

        monkeypatch.setattr(r, "create_and_start", create)
        return r

    def now(self):
        return datetime.datetime.now().timestamp()

    def test_healthy_analyzer_is_left_alone(self, runner):
        a = FakeAnalyzer(streaming_ts=self.now() - 10, last_data_ts=self.now())
        runner.analyzers = [a]
        runner.check_analyzers()
        assert runner.analyzers == [a]
        assert not a.terminated

    def test_analyzer_in_startup_grace_is_left_alone(self, runner):
        a = FakeAnalyzer(started_at=self.now() - 5)
        runner.analyzers = [a]
        runner.check_analyzers()
        assert runner.analyzers == [a]

    def test_analyzer_streaming_without_data_yet_is_left_alone(self, runner):
        a = FakeAnalyzer(streaming_ts=self.now() - 1)
        runner.analyzers = [a]
        runner.check_analyzers()
        assert runner.analyzers == [a]

    def test_analyzer_that_never_starts_streaming_is_restarted(self, runner):
        a = FakeAnalyzer(started_at=self.now() - 60)
        runner.analyzers = [a]
        runner.check_analyzers()
        assert a.terminated
        assert runner.analyzers == runner.created
        assert runner.created[0].sdr_max_restart == 2
        assert a.signal_queue.get_nowait().state is StateMessage.State.STOPPED

    def test_analyzer_streaming_but_never_receiving_is_restarted(self, runner):
        a = FakeAnalyzer(streaming_ts=self.now() - 30)
        runner.analyzers = [a]
        runner.check_analyzers()
        assert a.terminated and len(runner.created) == 1

    def test_stalled_analyzer_is_restarted(self, runner):
        a = FakeAnalyzer(streaming_ts=self.now() - 60, last_data_ts=self.now() - 30)
        runner.analyzers = [a]
        runner.check_analyzers()
        assert a.terminated
        assert runner.analyzers == runner.created

    def test_dead_analyzer_is_restarted(self, runner):
        a = FakeAnalyzer(alive=False)
        runner.analyzers = [a]
        runner.check_analyzers()
        assert runner.analyzers == runner.created
        assert runner.created[0].device == a.device

    def test_restart_budget_is_exhausted(self, runner, monkeypatch):
        terminated = []
        monkeypatch.setattr(runner, "terminate", terminated.append)
        runner.analyzers = [FakeAnalyzer(alive=False, max_restart=0)]
        runner.check_analyzers()
        assert terminated == [signal.SIGTERM]
        assert runner.created == []

    def test_only_the_failed_analyzer_is_replaced(self, runner):
        ok = FakeAnalyzer("0", streaming_ts=self.now() - 10, last_data_ts=self.now())
        dead = FakeAnalyzer("1", alive=False)
        runner.analyzers = [ok, dead]
        runner.check_analyzers()
        assert runner.analyzers[0] is ok
        assert [a.device for a in runner.analyzers] == ["0", "1"]
        assert runner.analyzers[1] is runner.created[0]


class TestStopAndTerminate:
    def test_stop_analyzers_reports_stopped_and_clears(self, make_runner):
        r = make_runner()
        a = FakeAnalyzer()
        r.analyzers = [a]
        r.stop_analyzers()
        assert a.terminated
        assert r.analyzers == []
        assert a.signal_queue.get_nowait().state is StateMessage.State.STOPPED

    def test_unresponsive_analyzer_is_killed(self, make_runner):
        r = make_runner()
        a = FakeAnalyzer()
        a.terminate = lambda: None  # ignores SIGTERM
        r.analyzers = [a]
        r.stop_analyzers()
        assert a.killed

    def test_full_queue_does_not_block(self, make_runner):
        r = make_runner()
        a = FakeAnalyzer()
        a.signal_queue = queue.Queue(maxsize=1)
        a.signal_queue.put("x")
        r.analyzers = [a]
        r.stop_analyzers()
        assert a.terminated

    def test_terminate_stops_main_loop_and_analyzers(self, make_runner):
        r = make_runner()
        a = FakeAnalyzer()
        r.analyzers = [a]
        r.terminate(signal.SIGTERM)
        assert not r.running
        assert a.terminated
        assert r._terminate_in_progress is False

    def test_nested_terminate_is_ignored(self, make_runner):
        r = make_runner()
        r._terminate_in_progress = True
        a = FakeAnalyzer()
        r.analyzers = [a]
        r.terminate(signal.SIGINT)
        assert not a.terminated
        assert r.running


class TestStartAnalyzers:
    def test_one_analyzer_per_device(self, make_runner, monkeypatch):
        r = make_runner("-d", "0", "1", "-c", "1.0", "2.0")
        calls = []
        monkeypatch.setattr(r, "create_and_start", lambda d, c: calls.append((d, c)) or FakeAnalyzer(d))
        r.start_analyzers()
        assert calls == [("0", 1.0), ("1", 2.0)]
        assert [a.device for a in r.analyzers] == ["0", "1"]

    def test_main_starts_the_analyzers(self, make_runner, monkeypatch):
        r = make_runner()
        started = []
        monkeypatch.setattr(r, "start_analyzers", lambda: started.append(1))
        monkeypatch.setattr(r, "check_analyzers", lambda: None)
        monkeypatch.setattr(r.connector, "step", lambda timeout: setattr(r, "running", False))
        with pytest.raises(SystemExit) as exc:
            r.main()
        assert exc.value.code == 0
        assert started == [1]

    def test_main_flushes_matches_while_idle_and_on_exit(self, make_runner, monkeypatch):
        r = make_runner()
        monkeypatch.setattr(r, "start_analyzers", lambda: None)
        monkeypatch.setattr(r, "check_analyzers", lambda: None)
        calls = []
        monkeypatch.setattr(r.matcher, "flush", lambda **kw: calls.append(kw))
        steps = iter([True, False])  # afterwards: idle once more, and stop

        def step(timeout):
            try:
                return next(steps)
            except StopIteration:
                r.running = False
                return False

        monkeypatch.setattr(r.connector, "step", step)
        with pytest.raises(SystemExit):
            r.main()
        # not after the busy step, after each idle step, forced at the end
        assert calls == [{}, {}, {"force": True}]

    def test_pending_matches_are_published_on_exit(self, make_runner, make_signal, monkeypatch):
        r = make_runner()
        monkeypatch.setattr(r, "start_analyzers", lambda: None)
        monkeypatch.setattr(r, "check_analyzers", lambda: None)
        published = []
        r.connector.consumers.append(SimpleNamespace(add=published.append))
        r.connector.q.put(make_signal())
        r.running = True
        original = r.connector.step
        # the main loop passes a possibly negative timeout, which would race with the queue's feeder thread
        monkeypatch.setattr(
            r.connector, "step", lambda t: (setattr(r, "running", False), original(datetime.timedelta(seconds=0.5)))[1]
        )
        with pytest.raises(SystemExit):
            r.main()
        assert [type(m).__name__ for m in published] == ["Signal", "MatchingSignal"]


def test_parser_has_no_dashboard_or_schedule_options():
    help_text = Runner.parser.format_help()
    assert "dashboard" not in help_text
    assert "--schedule" not in help_text
