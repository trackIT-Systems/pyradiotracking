import datetime
import queue
import time

import pytest

from radiotracking import CalibratedMatchedSignal, MatchingSignal, Signal, StateMessage
from radiotracking.match import SignalMatcher

from .conftest import T0


def ms(n):
    return datetime.timedelta(milliseconds=n)


@pytest.fixture
def out():
    return queue.Queue()


@pytest.fixture
def matcher(out):
    return SignalMatcher(
        device=["0", "1", "2", "3"],
        matching_timeout_s=2.0,
        matching_time_diff_s=0.005,
        matching_bandwidth_hz=1000,
        signal_queue=out,
    )


def drain(q):
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


def test_signals_of_all_devices_are_grouped(matcher, make_signal, out):
    for dev in "0123":
        matcher.add(make_signal(device=dev, ts=T0 + ms(int(dev)), avg_dbw=-50 - int(dev)))
    assert len(matcher._matched) == 1
    assert matcher._matched[0]._avgs == [-50, -51, -52, -53]
    assert drain(out) == []


def test_distant_signals_are_not_grouped(matcher, make_signal):
    matcher.add(make_signal(device="0"))
    matcher.add(make_signal(device="1", ts=T0 + ms(500)))
    matcher.add(make_signal(device="2", frequency=150_500_000))
    assert len(matcher._matched) == 3


def test_timed_out_match_is_emitted_by_next_signal(matcher, make_signal, out):
    matcher.add(make_signal(device="0"))
    matcher.add(make_signal(device="1", ts=T0 + datetime.timedelta(seconds=3), frequency=151_000_000))
    emitted = drain(out)
    assert len(emitted) == 1
    assert isinstance(emitted[0], MatchingSignal)
    assert emitted[0]._avgs == [-55, None, None, None]
    assert len(matcher._matched) == 1


def test_non_signals_are_ignored(matcher, out):
    matcher.add(StateMessage("0", T0, StateMessage.State.RUNNING))
    assert matcher._matched == []
    assert drain(out) == []


def test_duplicate_device_keeps_the_stronger_signal(matcher, make_signal):
    matcher.add(make_signal(device="0", avg_dbw=-60))
    matcher.add(make_signal(device="0", avg_dbw=-50))
    assert len(matcher._matched) == 1
    assert matcher._matched[0]._avgs[0] == -50


def test_duration_diff_is_optional(out, make_signal):
    m = SignalMatcher(["0", "1"], 2.0, 0.005, 1000, out, matching_duration_diff_ms=2)
    m.add(make_signal(device="0", duration=0.02))
    m.add(make_signal(device="1", duration=0.04))
    assert len(m._matched) == 2


class TestFlush:
    def test_flushes_timed_out_matches_without_further_signals(self, matcher, make_signal, out):
        matcher.add(make_signal(device="0"))
        matcher.flush(now=time.monotonic() + 10)
        emitted = drain(out)
        assert len(emitted) == 1
        assert emitted[0]._avgs == [-55, None, None, None]
        assert matcher._matched == []

    def test_keeps_matches_that_can_still_grow(self, matcher, make_signal, out):
        matcher.add(make_signal(device="0"))
        matcher.flush()
        assert drain(out) == []
        assert len(matcher._matched) == 1

    def test_only_old_matches_are_flushed(self, matcher, make_signal, out):
        matcher.add(make_signal(device="0"))
        matcher._matched[0]._created -= 5
        matcher.add(make_signal(device="1", frequency=151_000_000))
        matcher.flush()
        assert len(drain(out)) == 1
        assert len(matcher._matched) == 1

    def test_force_flushes_everything_in_order(self, matcher, make_signal, out):
        matcher.add(make_signal(device="0", frequency=150_000_000))
        matcher.add(make_signal(device="0", frequency=151_000_000))
        matcher.flush(force=True)
        assert [m.frequency for m in drain(out)] == [150_000_000, 151_000_000]
        assert matcher._matched == []

    def test_flush_on_empty_matcher(self, matcher, out):
        matcher.flush()
        matcher.flush(force=True)
        assert drain(out) == []

    def test_a_flushed_match_is_not_extended(self, matcher, make_signal, out):
        matcher.add(make_signal(device="0"))
        matcher.flush(force=True)
        matcher.add(make_signal(device="1"))
        assert len(matcher._matched) == 1
        assert matcher._matched[0]._avgs == [None, -55, None, None]


class TestMatchingEdges:
    @pytest.mark.parametrize("offset,grouped", [(500, True), (-500, True), (501, False), (-501, False)])
    def test_frequency_margin_is_half_the_bandwidth_to_each_side(self, matcher, make_signal, offset, grouped):
        matcher.add(make_signal(device="0"))
        matcher.add(make_signal(device="1", frequency=150_100_000.0 + offset))
        assert len(matcher._matched) == (1 if grouped else 2)

    @pytest.mark.parametrize("gap_ms,grouped", [(5, True), (6, False)])
    def test_time_margin_applies_between_end_and_start(self, matcher, make_signal, gap_ms, grouped):
        # the first signal lasts 20 ms, the second starts gap_ms after its end
        matcher.add(make_signal(device="0", duration=0.02))
        matcher.add(make_signal(device="1", ts=T0 + ms(20 + gap_ms), duration=0.02))
        assert len(matcher._matched) == (1 if grouped else 2)

    def test_members_arriving_out_of_order(self, matcher, make_signal):
        # the analyzers run independently, a later detection may be queued first
        matcher.add(make_signal(device="1", ts=T0 + ms(2)))
        matcher.add(make_signal(device="0", ts=T0))
        assert len(matcher._matched) == 1
        assert matcher._matched[0].ts == T0

    def test_simultaneous_tags_on_different_frequencies_are_separate(self, matcher, make_signal):
        for dev in "01":
            matcher.add(make_signal(device=dev, frequency=150_100_000.0, avg_dbw=-50))
            matcher.add(make_signal(device=dev, frequency=150_120_000.0, avg_dbw=-70))
        assert len(matcher._matched) == 2
        assert sorted(m._avgs[:2] for m in matcher._matched) == [[-70, -70], [-50, -50]]

    def test_next_pulse_of_the_same_tag_is_a_new_match(self, matcher, make_signal):
        matcher.add(make_signal(device="0"))
        matcher.add(make_signal(device="0", ts=T0 + datetime.timedelta(seconds=1)))
        assert len(matcher._matched) == 2

    def test_signal_of_an_unconfigured_device_is_not_in_the_output(self, matcher, make_signal):
        # characterization: the device names of the analyzers and the matcher have to be the same strings
        matcher.add(make_signal(device="0"))
        matcher.add(make_signal(device="7"))
        assert len(matcher._matched) == 1
        assert matcher._matched[0]._avgs == [-55.0, None, None, None]


def test_calibrated_match_is_published_after_the_plain_one():
    q = queue.Queue()
    m = SignalMatcher(["0", "1"], 1.0, 0.1, 1000.0, q)
    m.add(Signal("0", T0, 150_100_000.0, 0.02, -50, -55, 1.0, -80, 25, -90.0, -95.0, -115.0))
    m.flush(force=True)
    first, second = q.get_nowait(), q.get_nowait()
    assert isinstance(first, MatchingSignal)
    assert isinstance(second, CalibratedMatchedSignal)
    assert second.as_list[3:] == [-95.0, None]


def test_uncalibrated_match_has_no_calibrated_message(make_signal):
    q = queue.Queue()
    m = SignalMatcher(["0", "1"], 1.0, 0.1, 1000.0, q)
    m.add(make_signal())
    m.flush(force=True)
    assert isinstance(q.get_nowait(), MatchingSignal)
    assert q.empty()
