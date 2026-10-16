import datetime

import numpy as np
import pytest

from radiotracking import MatchedSignal, MatchingSignal, Signal, StateMessage, dB, from_dB

from .conftest import T0


def test_db_roundtrip():
    assert dB(1.0) == 0
    assert dB(100.0) == pytest.approx(20)
    assert from_dB(30) == pytest.approx(1000)
    assert from_dB(dB(0.123)) == pytest.approx(0.123)
    assert dB(np.array([1.0, 10.0])).tolist() == pytest.approx([0, 10])


class TestSignal:
    def test_parses_strings(self):
        sig = Signal("0", T0.isoformat(), "150100000", "0.02", "-50", "-55", "1.5", "-80", "25")
        assert sig.ts == T0
        assert sig.frequency == 150_100_000.0
        assert sig.duration == datetime.timedelta(milliseconds=20)
        assert (sig.max, sig.avg, sig.std, sig.noise, sig.snr) == (-50, -55, 1.5, -80, 25)

    def test_accepts_native_types(self, make_signal):
        sig = make_signal(duration=datetime.timedelta(milliseconds=12))
        assert sig.duration == datetime.timedelta(milliseconds=12)

    def test_as_dict_follows_header(self, make_signal):
        sig = make_signal()
        assert len(sig.header) == len(sig.as_list)
        assert list(sig.as_dict) == sig.header
        assert sig.as_dict["Device"] == "0"

    def test_str_and_repr(self, make_signal):
        sig = make_signal()
        assert "150.100 MHz" in str(sig)
        assert "20.00 ms" in str(sig)
        assert repr(sig).startswith("Signal(")


class TestStateMessage:
    @pytest.mark.parametrize("raw", [1, "1", StateMessage.State.RUNNING])
    def test_state_parsing(self, raw):
        assert StateMessage("0", T0, raw).state is StateMessage.State.RUNNING

    def test_invalid_state(self):
        with pytest.raises(ValueError):
            StateMessage("0", T0, 7)

    def test_as_list_uses_state_value(self):
        msg = StateMessage("0", T0, StateMessage.State.STARTED, peak_db=-10.0)
        assert msg.as_list == ["0", T0, 2, -10.0, None, None]
        assert list(msg.as_dict) == StateMessage.header


class TestMatchedSignal:
    def test_header_contains_devices(self):
        msig = MatchedSignal(["0", "1"], T0, 150e6, 0.02, -50, -60)
        assert msig.header == ["Time", "Frequency", "Duration", "0", "1"]
        assert msig.as_list == [T0, 150e6, datetime.timedelta(milliseconds=20), -50.0, -60.0]

    def test_missing_avg_is_none(self):
        msig = MatchedSignal(["0", "1"], T0, 150e6, 0.02, -50, None)
        assert msig._avgs == [-50.0, None]

    def test_str_with_missing_avg(self):
        msig = MatchedSignal(["0", "1"], T0, 150e6, 0.02, -50, None)
        assert "-50.00" in str(msig)
        assert "None" in str(msig)

    @pytest.mark.xfail(strict=True, reason="__str__ tests `if avg`, so a power of exactly 0.0 dBW is printed as None")
    def test_str_with_zero_avg(self):
        msig = MatchedSignal(["0"], T0, 150e6, 0.02, 0.0)
        assert "None" not in str(msig)


class TestMatchingSignal:
    def test_aggregates_members(self, make_signal):
        msig = MatchingSignal(["0", "1", "2"])
        msig.add_member(make_signal(device="0", ts=T0 + datetime.timedelta(milliseconds=3), frequency=150_000_000, duration=0.01, avg_dbw=-60))
        msig.add_member(make_signal(device="2", ts=T0, frequency=150_000_400, duration=0.03, avg_dbw=-40))
        assert msig.ts == T0
        assert msig.duration == datetime.timedelta(milliseconds=30)
        assert msig.frequency == 150_000_200
        assert msig._avgs == [-60, None, -40]
        assert msig.header[3:] == ["0", "1", "2"]
        assert len(msig.as_list) == len(msig.header)

    def test_stronger_duplicate_replaces(self, make_signal):
        msig = MatchingSignal(["0"])
        msig.add_member(make_signal(avg_dbw=-60))
        msig.add_member(make_signal(avg_dbw=-50))
        msig.add_member(make_signal(avg_dbw=-70))
        assert msig._avgs == [-50]

    def test_empty_matching_signal_header_works(self):
        # the header is written to the CSV before any signal exists
        assert MatchingSignal(["0", "1"]).header == ["Time", "Frequency", "Duration", "0", "1"]

    def test_has_member_frequency(self, make_signal):
        msig = MatchingSignal(["0", "1"])
        msig.add_member(make_signal(frequency=150_000_000))
        assert msig.has_member(make_signal(device="1", frequency=150_000_000))
        assert not msig.has_member(make_signal(device="1", frequency=150_000_100))
        assert msig.has_member(make_signal(device="1", frequency=150_000_100), bandwidth=400)
        assert not msig.has_member(make_signal(device="1", frequency=150_000_300), bandwidth=400)

    def test_has_member_time(self, make_signal):
        msig = MatchingSignal(["0", "1"])
        msig.add_member(make_signal(duration=0.02))
        later = lambda ms: T0 + datetime.timedelta(milliseconds=ms)  # noqa: E731
        assert msig.has_member(make_signal(device="1", ts=later(15)))
        assert not msig.has_member(make_signal(device="1", ts=later(30)))
        assert msig.has_member(make_signal(device="1", ts=later(30)), time_diff=datetime.timedelta(milliseconds=15))
        assert not msig.has_member(make_signal(device="1", ts=later(-30), duration=0.01))
        assert msig.has_member(
            make_signal(device="1", ts=later(-30), duration=0.01), time_diff=datetime.timedelta(milliseconds=25)
        )

    def test_has_member_duration(self, make_signal):
        msig = MatchingSignal(["0", "1"])
        msig.add_member(make_signal(duration=0.02))
        diff = datetime.timedelta(milliseconds=4)
        assert msig.has_member(make_signal(device="1", duration=0.021), duration_diff=diff)
        assert not msig.has_member(make_signal(device="1", duration=0.03), duration_diff=diff)
        assert not msig.has_member(make_signal(device="1", duration=0.01), duration_diff=diff)
        # no duration_diff: durations are not compared
        assert msig.has_member(make_signal(device="1", duration=0.03))
