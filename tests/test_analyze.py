import datetime
import multiprocessing
import queue
import signal

import numpy as np
import pytest
import scipy.signal

from radiotracking import Signal, StateMessage, dB
from radiotracking.analyze import SignalAnalyzer, default_analysis_block_samples

from .conftest import T0

FS = 300_000
NPERSEG = 256
CENTER = 150_150_000
BIN_HZ = FS / NPERSEG
TONE_BIN = 43  # ~ +50.4 kHz


def make_analyzer(**over):
    kw = dict(
        device="0",
        calibration_db=0.0,
        sample_rate=FS,
        center_freq=CENTER,
        gain=49.6,
        lna_gain=15,
        mixer_gain=14,
        vga_gain=7,
        tuner_bandwidth=0,
        fft_nperseg=NPERSEG,
        fft_window="hamming",
        signal_min_duration_ms=8,
        signal_max_duration_ms=40,
        signal_threshold_dbw=-100,
        snr_threshold_db=5,
        verbose=0,
        sdr_max_restart=3,
        sdr_timeout_s=2,
        state_update_s=300,
        sdr_callback_length=None,
        analysis_block_samples=None,
        signal_queue=queue.Queue(),
        last_data_ts=multiprocessing.Value("d", 0.0),
        streaming_ts=multiprocessing.Value("d", 0.0),
        status_interval=0,
    )
    kw.update(over)
    return SignalAnalyzer(**kw)


def burst_iq(n_samples, start_s, dur_s, tone_bin=TONE_BIN, amp=1.0, noise=1e-3, seed=0):
    """Complex noise with a tone burst of `dur_s` starting at `start_s`."""
    rng = np.random.default_rng(seed)
    iq = noise * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
    a, b = int(start_s * FS), int((start_s + dur_s) * FS)
    t = np.arange(a, min(b, n_samples)) / FS
    iq[a : min(b, n_samples)] += amp * np.exp(2j * np.pi * tone_bin * BIN_HZ * t)
    return iq.astype(np.complex64)


def spectrogram(an, iq):
    return scipy.signal.spectrogram(
        iq, fs=an.sample_rate, window=an.fft_window, nperseg=an.fft_nperseg, noverlap=0, return_onesided=False
    )


def detect(an, iq, ts_start=T0):
    freqs, times, spec = spectrogram(an, iq)
    sigs = an.extract_signals(freqs, times, spec, ts_start)
    return an.filter_shadow_signals(sigs), spec


def test_default_analysis_block_samples():
    assert default_analysis_block_samples(FS, NPERSEG) == 118 * NPERSEG
    assert default_analysis_block_samples(FS, NPERSEG) % NPERSEG == 0
    assert default_analysis_block_samples(1000, 256) == 256
    assert default_analysis_block_samples(FS, 300_000) == 300_000
    with pytest.raises(ValueError):
        default_analysis_block_samples(FS, 0)


@pytest.mark.parametrize("n,expected", [(1, 256), (256, 256), (257, 512), (1000, 1024)])
def test_analysis_block_alignment(n, expected):
    assert make_analyzer()._align_block_to_fft(n) == expected


def test_configured_block_size_wins():
    an = make_analyzer(analysis_block_samples=1000)
    assert an._finalize_analysis_block_samples() == 1024
    assert make_analyzer()._finalize_analysis_block_samples() == 118 * NPERSEG


def test_device_must_be_index_or_known_serial():
    assert make_analyzer(device="2").device_index == 2
    with pytest.raises(SystemExit):
        make_analyzer(device="not-a-serial")


def test_thresholds_include_calibration():
    an = make_analyzer(signal_threshold_dbw=-90, calibration_db=10, snr_threshold_db=3)
    assert dB(an.signal_threshold) == pytest.approx(-80)
    assert dB(an.snr_threshold) == pytest.approx(3)


class TestGain:
    def test_total_gain_matches_librtlsdr_tables(self):
        an = make_analyzer(lna_gain=15, mixer_gain=14, vga_gain=7)
        assert an._r82xx_get_lna_gain() == pytest.approx(33.5)
        assert an._r82xx_get_mixer_gain() == pytest.approx(16.1)
        assert an._r82xx_get_vga_gain() == pytest.approx(19.6)
        assert an._r82xx_get_gain() == pytest.approx(64.5)

    def test_minimum_gain(self):
        an = make_analyzer(lna_gain=0, mixer_gain=0, vga_gain=0)
        assert an._r82xx_get_gain() == pytest.approx(-4.7)

    @pytest.mark.parametrize("field,getter", [("lna_gain", "lna"), ("mixer_gain", "mixer"), ("vga_gain", "vga")])
    @pytest.mark.parametrize("value", [-1, 16])
    def test_out_of_range_index(self, field, getter, value):
        an = make_analyzer()
        setattr(an, field, value)
        with pytest.raises(ValueError):
            getattr(an, f"_r82xx_get_{getter}_gain")()

    @pytest.mark.xfail(strict=True, reason="ValueError is raised with (format, arg) instead of a formatted message")
    def test_error_message_is_formatted(self):
        an = make_analyzer(lna_gain=16)
        with pytest.raises(ValueError) as exc:
            an._r82xx_get_lna_gain()
        assert "%s" not in str(exc.value)


class TestRingBuffer:
    @pytest.fixture
    def an(self):
        an = make_analyzer()
        an._ring_capacity = 10
        an._ring_buffer = np.empty(10, dtype=np.complex64)
        return an

    def test_push_pop_fifo(self, an):
        assert an._ring_push(np.arange(4, dtype=np.complex64))
        assert an._ring_count == 4
        assert an._ring_pop(3).real.tolist() == [0, 1, 2]
        assert an._ring_count == 1

    def test_wraparound(self, an):
        an._ring_push(np.arange(8, dtype=np.complex64))
        an._ring_pop(6)
        assert an._ring_push(np.arange(10, 16, dtype=np.complex64))  # wraps
        assert an._ring_write == 4
        assert an._ring_pop(8).real.tolist() == [6, 7, 10, 11, 12, 13, 14, 15]
        assert an._ring_count == 0

    def test_overflow_is_rejected_without_modification(self, an):
        an._ring_push(np.zeros(8, dtype=np.complex64))
        assert not an._ring_push(np.zeros(3, dtype=np.complex64))
        assert an._ring_count == 8

    def test_exact_fill(self, an):
        assert an._ring_push(np.ones(10, dtype=np.complex64))
        assert an._ring_pop(10).real.tolist() == [1] * 10

    def test_converts_dtype(self, an):
        an._ring_push(np.array([1 + 2j], dtype=np.complex128))
        assert an._ring_pop(1)[0] == 1 + 2j


class TestUpdateState:
    @pytest.fixture
    def an(self):
        an = make_analyzer(state_update_s=300)
        an.last_state = None
        return an

    def states(self, an):
        out = []
        while not an.signal_queue.empty():
            out.append(an.signal_queue.get_nowait())
        return out

    def test_same_state_is_throttled(self, an):
        an.update_state(T0, StateMessage.State.RUNNING)
        an.update_state(T0 + datetime.timedelta(seconds=10), StateMessage.State.RUNNING)
        assert len(self.states(an)) == 1

    def test_same_state_repeats_after_interval(self, an):
        an.update_state(T0, StateMessage.State.RUNNING)
        an.update_state(T0 + datetime.timedelta(seconds=301), StateMessage.State.RUNNING)
        assert len(self.states(an)) == 2

    def test_state_change_is_sent_immediately(self, an):
        an.update_state(T0, StateMessage.State.STARTED)
        an.update_state(T0 + datetime.timedelta(seconds=1), StateMessage.State.RUNNING)
        assert [s.state for s in self.states(an)] == [StateMessage.State.STARTED, StateMessage.State.RUNNING]

    def test_metrics_bypass_throttling(self, an):
        an.update_state(T0, StateMessage.State.RUNNING)
        an.update_state(T0, StateMessage.State.RUNNING, peak_db=-10.0, rms_db=-40.0, snr_db=30.0)
        sent = self.states(an)
        assert len(sent) == 2
        assert sent[1].peak_db == -10.0

    def test_timestamps_are_timezone_aware(self, an):
        an.update_state(datetime.datetime(2026, 1, 1, 12), StateMessage.State.RUNNING)
        assert self.states(an)[0].ts.tzinfo is not None


class TestHandleSignal:
    def test_sigterm_stops_the_consumer_and_the_sdr(self):
        an = make_analyzer()
        an.last_state = None
        an.sdr = type("Sdr", (), {"cancel_read_async": lambda self: setattr(self, "cancelled", True)})()
        import threading

        an._consumer_stop = threading.Event()
        an._ring_cv = threading.Condition()
        an.handle_signal(signal.SIGTERM, None)
        assert an._consumer_stop.is_set()
        assert an.sdr.cancelled
        assert an.signal_queue.get_nowait().state is StateMessage.State.STOPPED

    def test_sigint_is_ignored(self):
        an = make_analyzer()
        an.handle_signal(signal.SIGINT, None)
        assert an.signal_queue.empty()

    @pytest.mark.xfail(
        strict=True,
        reason="self.sdr is only assigned after the SDR was opened, but the handlers are installed at the top of run(): a SIGTERM while opening the device (the hang case the watchdog handles) raises AttributeError",
    )
    def test_sigterm_before_the_sdr_is_open(self):
        an = make_analyzer()
        an.last_state = None
        an.handle_signal(signal.SIGTERM, None)


class TestShadowFilter:
    def test_louder_overlapping_signal_shadows(self, make_signal):
        loud = make_signal(frequency=150_000_000, max_dbw=-40)
        weak = make_signal(frequency=150_001_000, max_dbw=-60)
        assert SignalAnalyzer.is_shadow_of(weak, [loud, weak]) == 0
        assert SignalAnalyzer.is_shadow_of(loud, [loud, weak]) is None

    def test_non_overlapping_signals_are_independent(self, make_signal):
        a = make_signal(ts=T0, duration=0.01, max_dbw=-40)
        b = make_signal(ts=T0 + datetime.timedelta(seconds=1), duration=0.01, max_dbw=-60)
        assert SignalAnalyzer.is_shadow_of(b, [a, b]) is None

    def test_equal_power_signals_both_survive(self, make_signal):
        a = make_signal(frequency=150_000_000)
        b = make_signal(frequency=150_001_000)
        assert make_analyzer().filter_shadow_signals([a, b]) == [a, b]

    def test_filter_keeps_strongest_of_cluster(self, make_signal):
        sigs = [make_signal(max_dbw=m, frequency=150_000_000 + i * 1000) for i, m in enumerate([-60, -40, -55])]
        assert make_analyzer().filter_shadow_signals(sigs) == [sigs[1]]

    def test_empty(self):
        assert make_analyzer().filter_shadow_signals([]) == []

    def test_frequency_is_not_considered(self, make_signal):
        # characterization: a weaker signal is dropped whenever a louder one overlaps in time,
        # even if it is 100 kHz away, i.e. two transmitters sending at the same time are not both detected
        loud = make_signal(frequency=150_000_000, max_dbw=-40)
        weak = make_signal(frequency=150_100_000, max_dbw=-60)
        assert make_analyzer().filter_shadow_signals([loud, weak]) == [loud]


class TestExtractSignals:
    def test_noise_only_yields_nothing(self):
        an = make_analyzer()
        sigs, _ = detect(an, burst_iq(30208, 0.0, 0.0))
        assert sigs == []

    def test_detects_burst(self):
        an = make_analyzer()
        sigs, _ = detect(an, burst_iq(30208, 0.03, 0.02))
        assert len(sigs) == 1
        sig = sigs[0]
        assert isinstance(sig, Signal)
        assert sig.frequency == pytest.approx(CENTER + TONE_BIN * BIN_HZ, abs=1)
        assert sig.duration.total_seconds() == pytest.approx(0.02, abs=0.003)
        assert (sig.ts - T0).total_seconds() == pytest.approx(0.03, abs=0.003)
        assert sig.snr > 5
        assert sig.max >= sig.avg

    def test_negative_frequency_offset(self):
        an = make_analyzer()
        sigs, _ = detect(an, burst_iq(30208, 0.03, 0.02, tone_bin=-30))
        assert len(sigs) == 1
        assert sigs[0].frequency == pytest.approx(CENTER - 30 * BIN_HZ, abs=1)

    def test_calibration_offsets_power(self):
        iq = burst_iq(30208, 0.03, 0.02)
        plain = detect(make_analyzer(), iq)[0][0]
        calibrated = detect(make_analyzer(calibration_db=6.0), iq)[0][0]
        assert plain.max - calibrated.max == pytest.approx(6.0)
        assert plain.snr == pytest.approx(calibrated.snr)

    def test_too_short_burst_is_ignored(self):
        sigs, _ = detect(make_analyzer(), burst_iq(30208, 0.03, 0.004))
        assert sigs == []

    def test_too_long_burst_is_ignored(self):
        sigs, _ = detect(make_analyzer(signal_max_duration_ms=40), burst_iq(30208, 0.02, 0.06))
        assert sigs == []

    def test_weak_signal_below_threshold_is_ignored(self):
        iq = burst_iq(30208, 0.03, 0.02, amp=1e-2, noise=1e-4)
        assert detect(make_analyzer(signal_threshold_dbw=-60), iq)[0] == []
        assert len(detect(make_analyzer(signal_threshold_dbw=-100), iq)[0]) == 1

    def test_two_tones_are_two_signals(self):
        iq = burst_iq(30208, 0.03, 0.02, tone_bin=TONE_BIN) + burst_iq(30208, 0.08, 0.02, tone_bin=-50, noise=0)
        sigs, _ = detect(make_analyzer(), iq)
        assert sorted(round((s.frequency - CENTER) / BIN_HZ) for s in sigs) == [-50, TONE_BIN]

    def test_burst_touching_the_end_waits_for_the_next_chunk(self):
        an = make_analyzer()
        block = 30208
        total = burst_iq(2 * block, (block / FS) - 0.01, 0.02)
        first, second = total[:block], total[block:]

        sigs1, spec1 = detect(an, first, T0)
        assert sigs1 == []  # laps into the next spectrogram
        an._spectrogram_last = spec1

        ts2 = T0 + datetime.timedelta(seconds=block / FS)
        sigs2, _ = detect(an, second, ts2)
        assert len(sigs2) == 1
        assert sigs2[0].duration.total_seconds() == pytest.approx(0.02, abs=0.004)
        assert (sigs2[0].ts - T0).total_seconds() == pytest.approx(block / FS - 0.01, abs=0.004)

    def test_signal_starting_in_the_first_sample_of_a_chunk(self):
        an = make_analyzer()
        sigs, _ = detect(an, burst_iq(30208, 0.0, 0.02))
        assert len(sigs) == 1
        assert sigs[0].duration.total_seconds() == pytest.approx(0.02, abs=0.003)

    @pytest.mark.xfail(
        strict=True,
        raises=IndexError,
        reason="the segment duration is taken from times[1] - times[0], which does not exist when analysis_block_samples == fft_nperseg",
    )
    def test_single_segment_block_does_not_crash(self):
        an = make_analyzer()
        iq = burst_iq(NPERSEG, 0.0, 0.0)
        freqs, times, spec = spectrogram(an, iq)
        assert len(times) == 1
        assert an.extract_signals(freqs, times, spec, T0) == []
