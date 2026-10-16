import datetime
import logging
import multiprocessing
import queue
import signal

import numpy as np
import pytest
import scipy.signal

from radiotracking import Signal, StateMessage, dB
from radiotracking.analyze import SignalAnalyzer, resolve_block_samples, warn_if_block_too_short

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


def burst_iq(n_samples, start_s, dur_s, tone_bin=TONE_BIN, amp=1.0, noise=1e-3, seed=0, fs=FS, bin_hz=BIN_HZ):
    """Complex noise with a tone burst of `dur_s` starting at `start_s`, the tone sits on an FFT bin (`bin_hz` wide)."""
    rng = np.random.default_rng(seed)
    iq = noise * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
    a, b = int(start_s * fs), int((start_s + dur_s) * fs)
    t = np.arange(a, min(b, n_samples)) / fs
    iq[a : min(b, n_samples)] += amp * np.exp(2j * np.pi * tone_bin * bin_hz * t)
    return iq.astype(np.complex64)


def spectrogram(an, iq):
    return scipy.signal.spectrogram(
        iq, fs=an.sample_rate, window=an.fft_window, nperseg=an.fft_nperseg, noverlap=0, return_onesided=False
    )


def detect(an, iq, ts_start=T0):
    freqs, times, spec = spectrogram(an, iq)
    sigs = an.extract_signals(freqs, times, spec, ts_start)
    return an.filter_shadow_signals(sigs), spec


@pytest.mark.parametrize(
    "sample_rate,nperseg,configured,expected",
    [
        (FS, 256, None, 90112),  # 0.3 s = 90000 samples, rounded up to 352 segments
        (1_200_000, 1024, None, 360448),  # same duration and segment length as the default setup
        (FS, 256, 30000, 30208),
        (FS, 256, 30208, 30208),
        (FS, 256, 1000, 1024),
        (FS, 256, 1, 256),  # at least one segment
        (1000, 256, None, 512),
    ],
)
def test_resolve_block_samples(sample_rate, nperseg, configured, expected):
    assert resolve_block_samples(sample_rate, nperseg, configured) == expected


def test_resolve_block_samples_rejects_an_invalid_segment_length():
    with pytest.raises(ValueError):
        resolve_block_samples(FS, 0)


def test_analyzer_resolves_its_block():
    assert make_analyzer()._finalize_analysis_block_samples() == 352 * NPERSEG
    assert make_analyzer(analysis_block_samples=1000)._finalize_analysis_block_samples() == 1024
    # the hardware may adjust the sample rate after the analyzer was created
    an = make_analyzer(sample_rate=250_000)
    an.sample_rate = 1_200_000
    assert an._finalize_analysis_block_samples() == resolve_block_samples(1_200_000, NPERSEG)


class TestBlockWarning:
    def test_default_block_is_long_enough_for_the_default_maximum_duration(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert not warn_if_block_too_short(resolve_block_samples(FS, NPERSEG), FS, 50)
        assert caplog.text == ""

    def test_exactly_six_times_is_fine(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert not warn_if_block_too_short(int(0.3 * FS), FS, 50)
        assert caplog.text == ""

    @pytest.mark.parametrize("block,max_ms", [(30208, 50), (int(0.29 * FS), 50), (90112, 60), (256, 8)])
    def test_short_block_warns(self, block, max_ms, caplog):
        with caplog.at_level(logging.WARNING):
            assert warn_if_block_too_short(block, FS, max_ms)
        assert "shorter than 6 times signal_max_duration_ms" in caplog.text
        assert "analysis_block_samples" in caplog.text

    def test_message_gives_the_numbers(self, caplog):
        with caplog.at_level(logging.WARNING):
            warn_if_block_too_short(30208, FS, 50)
        assert "30208 samples, 0.101 s" in caplog.text
        assert "50 ms, i.e. 0.300 s" in caplog.text

    def test_wide_band_block_is_judged_in_time(self, caplog):
        with caplog.at_level(logging.WARNING):
            # 30208 samples are 25 ms at 1.2 MHz
            assert warn_if_block_too_short(30208, 1_200_000, 50)
            assert not warn_if_block_too_short(resolve_block_samples(1_200_000, 1024), 1_200_000, 50)

    def test_analyzer_does_not_refuse_a_short_block(self):
        make_analyzer(analysis_block_samples=30208, signal_max_duration_ms=50)


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

    @pytest.mark.parametrize("field,getter", [("lna_gain", "lna"), ("mixer_gain", "mixer"), ("vga_gain", "vga")])
    def test_error_message_is_formatted(self, field, getter):
        an = make_analyzer()
        setattr(an, field, 16)
        with pytest.raises(ValueError) as exc:
            getattr(an, f"_r82xx_get_{getter}_gain")()
        assert str(exc.value).startswith(f"{field} index 16 not in 0 .. 15")


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

    def test_sigterm_before_the_sdr_is_open(self):
        an = make_analyzer()
        an.handle_signal(signal.SIGTERM, None)
        assert an._stop_requested
        assert an.signal_queue.get_nowait().state is StateMessage.State.STOPPED

    def test_sigalrm_before_the_sdr_is_open(self):
        an = make_analyzer()
        an.handle_signal(signal.SIGALRM, None)
        assert an._stop_requested


class TestRunTermination:
    @pytest.fixture(autouse=True)
    def restore_handlers(self):
        old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
        yield
        signal.alarm(0)
        for s, h in old.items():
            signal.signal(s, h)

    def test_terminated_while_opening_the_device_does_not_start_streaming(self, monkeypatch):
        import rtlsdr.rtlsdr

        an = make_analyzer()
        sdrs = []

        class FakeSdr:
            def __init__(self, index):
                self.sample_rate = FS
                self.closed = False
                self.streamed = False
                sdrs.append(self)
                # the watchdog terminates the process while the device is being opened
                an.handle_signal(signal.SIGTERM, None)

            def close(self):
                self.closed = True

            def read_samples_async(self, *args):
                self.streamed = True

        monkeypatch.setattr(rtlsdr.rtlsdr, "RtlSdr", FakeSdr)
        an.run()
        assert sdrs[0].closed
        assert not sdrs[0].streamed
        assert an.streaming_ts.value == 0.0

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
        iq = burst_iq(30208, 0.03, 0.02, tone_bin=TONE_BIN) + burst_iq(30208, 0.065, 0.02, tone_bin=-50, noise=0)
        sigs, _ = detect(make_analyzer(), iq)
        assert sorted(round((s.frequency - CENTER) / BIN_HZ) for s in sigs) == [-50, TONE_BIN]

    @pytest.mark.parametrize("dur_ms", [10, 20, 33, 40, 45])
    @pytest.mark.parametrize("amp,threshold", [(1.0, -100), (4e-4, -120)])
    def test_signals_of_10_to_45_ms_are_detected_in_the_default_block(self, dur_ms, amp, threshold):
        # amp 4e-4 is ~11 dB above the noise, 1.0 is ~80 dB; the SNR must not depend on how long the signal lasts
        an = make_analyzer(signal_max_duration_ms=50, signal_threshold_dbw=threshold)
        block = an._finalize_analysis_block_samples()
        sigs, _ = detect(an, burst_iq(block, 0.1, dur_ms / 1000, amp=amp))
        assert len(sigs) == 1
        assert sigs[0].duration.total_seconds() * 1000 == pytest.approx(dur_ms, abs=2)

    @pytest.mark.parametrize("dur_ms", [10, 20, 45])
    def test_noise_level_is_not_raised_by_the_signal(self, dur_ms):
        # expected noise power density of complex noise with variance 2 * noise**2 at FS: 2e-6 / FS
        an = make_analyzer(signal_max_duration_ms=50)
        block = an._finalize_analysis_block_samples()
        sigs, _ = detect(an, burst_iq(block, 0.1, dur_ms / 1000, amp=0.02))
        assert sigs[0].noise == pytest.approx(dB(2e-6 / FS), abs=1.2)
        assert sigs[0].snr > 40

    def test_strong_signal_does_not_hide_a_weak_one_in_another_bin(self):
        an = make_analyzer(signal_threshold_dbw=-120)
        block = an._finalize_analysis_block_samples()
        iq = burst_iq(block, 0.05, 0.02, tone_bin=TONE_BIN, amp=1.0) + burst_iq(block, 0.15, 0.02, tone_bin=-50, amp=4e-4, noise=0)
        sigs, _ = detect(an, iq)
        # the weak one is a shadow of nothing, as the strong one is not overlapping in time
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

    def test_single_segment_block_does_not_crash(self):
        # a block of one segment can not hold a signal (it always "laps into the next spectrogram")
        an = make_analyzer()
        iq = burst_iq(NPERSEG, 0.0, 0.0)
        freqs, times, spec = spectrogram(an, iq)
        assert len(times) == 1
        assert an.extract_signals(freqs, times, spec, T0) == []

    def test_single_segment_block_with_a_signal_does_not_crash(self):
        an = make_analyzer()
        freqs, times, spec = spectrogram(an, burst_iq(NPERSEG, 0.0, NPERSEG / FS))
        assert len(times) == 1
        assert an.extract_signals(freqs, times, spec, T0) == []

    @pytest.mark.parametrize("sample_rate,nperseg", [(300_000, 256), (1_200_000, 1024)])
    def test_wide_band_setup_behaves_like_the_default(self, sample_rate, nperseg):
        # 1.2 MHz with 1024 samples per segment has the same segment duration and bin width as the default,
        # and the default block is 0.3 s in both cases
        an = make_analyzer(sample_rate=sample_rate, fft_nperseg=nperseg)
        block = an._finalize_analysis_block_samples()
        assert block / sample_rate == pytest.approx(0.3, abs=0.002)
        assert nperseg / sample_rate == pytest.approx(256 / 300_000)

        iq = burst_iq(block, 0.03, 0.02, fs=sample_rate)
        sigs, _ = detect(an, iq)
        assert len(sigs) == 1
        assert sigs[0].frequency == pytest.approx(CENTER + TONE_BIN * BIN_HZ, abs=1)
        assert sigs[0].duration.total_seconds() == pytest.approx(0.02, abs=0.003)
        assert (sigs[0].ts - T0).total_seconds() == pytest.approx(0.03, abs=0.003)

    def test_segment_duration_follows_the_configured_sample_rate(self):
        # the hardware may adjust the sample rate after the analyzer was created
        an = make_analyzer(sample_rate=250_000)
        an.sample_rate = 300_000
        sigs, _ = detect(an, burst_iq(30208, 0.03, 0.02))
        assert len(sigs) == 1


@pytest.mark.parametrize("method", [m for m in ("fork", "spawn", "forkserver") if m in multiprocessing.get_all_start_methods()])
def test_analyzer_process_starts_with_every_start_method(method):
    """Python 3.14 changed the default start method on Linux to forkserver, the process object has to be picklable."""
    previous = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method(method, force=True)
    try:
        # device 99 does not exist, the child exits with an error right after it was started
        an = make_analyzer(
            device="99",
            signal_queue=multiprocessing.Queue(),
            last_data_ts=multiprocessing.Value("d", 0.0),
            streaming_ts=multiprocessing.Value("d", 0.0),
        )
        an.start()
        an.join(timeout=60)
        assert an.exitcode is not None
        assert an.exitcode == 1
    finally:
        multiprocessing.set_start_method(previous, force=True)


SEG_S = NPERSEG / FS
US = 1e-6  # timedelta resolution


def seg_s(n):
    """Time of the start of segment `n`, nudged so that burst_iq's int() lands exactly on the segment's first sample."""
    return (n * NPERSEG + 0.5) / FS


def detect_blocks(an, iq, block):
    """Run consecutive blocks through the analyzer like the consumer loop does, returns all filtered signals."""
    found = []
    for i in range(len(iq) // block):
        sigs, spec = detect(an, iq[i * block : (i + 1) * block], T0 + datetime.timedelta(seconds=i * block / FS))
        an._spectrogram_last = spec
        found += sigs
    return found


class TestSegmentPrecision:
    """A burst covering whole segments is reported with exactly these segments, no neighbouring noise segment."""

    @pytest.mark.parametrize("start,length", [(0, 24), (1, 24), (40, 10), (40, 11), (80, 30), (5, 12)])
    def test_start_and_duration_are_exact(self, start, length):
        sigs, _ = detect(make_analyzer(), burst_iq(30208, seg_s(start), length * SEG_S))
        assert len(sigs) == 1
        assert (sigs[0].ts - T0).total_seconds() == pytest.approx(start * SEG_S, abs=US)
        assert sigs[0].duration.total_seconds() == pytest.approx(length * SEG_S, abs=US)

    def test_constant_tone_has_no_power_deviation(self):
        # a noise segment counted into the signal raised the deviation to ~20 dB
        sigs, _ = detect(make_analyzer(), burst_iq(30208, seg_s(40), 24 * SEG_S))
        assert sigs[0].std == pytest.approx(0, abs=0.01)
        assert sigs[0].avg == pytest.approx(sigs[0].max, abs=0.01)

    @pytest.mark.parametrize("min_ms", [8, 10, 20])
    def test_burst_of_the_minimum_duration_is_found_at_every_offset(self, min_ms):
        # the search only seeds every int(min / segment) segments, a burst of the minimum duration must still be hit
        an = make_analyzer(signal_min_duration_ms=min_ms)
        length = int(np.ceil(min_ms / 1000 / SEG_S))
        for start in range(20, 20 + length + 1):
            sigs, _ = detect(an, burst_iq(30208, seg_s(start), length * SEG_S))
            assert len(sigs) == 1, f"burst at segment {start} missed"

    def test_minimum_duration_below_one_segment(self):
        an = make_analyzer(signal_min_duration_ms=0.1)
        sigs, _ = detect(an, burst_iq(30208, seg_s(30), 3 * SEG_S))
        assert len(sigs) == 1
        assert sigs[0].duration.total_seconds() == pytest.approx(3 * SEG_S, abs=US)

    def test_noise_level_includes_calibration(self):
        iq = burst_iq(30208, 0.03, 0.02)
        plain = detect(make_analyzer(), iq)[0][0]
        calibrated = detect(make_analyzer(calibration_db=6.0), iq)[0][0]
        assert plain.noise - calibrated.noise == pytest.approx(6.0)
        assert plain.avg - plain.noise == pytest.approx(calibrated.avg - calibrated.noise)


class TestBlockBoundaries:
    BLOCK = 30208
    NB = 30208 // NPERSEG  # 118 segments per block

    @pytest.mark.parametrize("pre", [1, 2, 10, 23])
    def test_burst_across_the_boundary_has_exact_start(self, pre):
        an = make_analyzer()
        sigs = detect_blocks(an, burst_iq(2 * self.BLOCK, seg_s(self.NB - pre), 24 * SEG_S), self.BLOCK)
        assert len(sigs) == 1
        assert (sigs[0].ts - T0).total_seconds() == pytest.approx((self.NB - pre) * SEG_S, abs=US)
        assert sigs[0].duration.total_seconds() == pytest.approx(24 * SEG_S, abs=US)
        assert sigs[0].std == pytest.approx(0, abs=0.01)

    def test_burst_ending_exactly_at_the_end_of_a_block_is_not_lost(self):
        an = make_analyzer()
        sigs = detect_blocks(an, burst_iq(2 * self.BLOCK, seg_s(self.NB - 24), 24 * SEG_S), self.BLOCK)
        assert len(sigs) == 1
        assert (sigs[0].ts - T0).total_seconds() == pytest.approx((self.NB - 24) * SEG_S, abs=US)
        assert sigs[0].duration.total_seconds() == pytest.approx(24 * SEG_S, abs=US)

    def test_burst_ending_one_segment_before_the_end_is_reported_once(self):
        an = make_analyzer()
        sigs = detect_blocks(an, burst_iq(2 * self.BLOCK, seg_s(self.NB - 25), 24 * SEG_S), self.BLOCK)
        assert len(sigs) == 1

    def test_burst_starting_exactly_at_a_block_is_reported_once(self):
        an = make_analyzer()
        sigs = detect_blocks(an, burst_iq(2 * self.BLOCK, seg_s(self.NB), 24 * SEG_S), self.BLOCK)
        assert len(sigs) == 1
        assert (sigs[0].ts - T0).total_seconds() == pytest.approx(self.NB * SEG_S, abs=US)

    def test_pulse_train_over_many_blocks(self):
        # a tag sending every 0.4 s, the pulses fall at every position relative to the blocks
        an = make_analyzer()
        n = 12 * self.BLOCK
        iq = sum(burst_iq(n, 0.013 + k * 0.4, 0.02, seed=k, noise=1e-3 if k == 0 else 0) for k in range(int(n / FS / 0.4)))
        sigs = detect_blocks(an, iq.astype(np.complex64), self.BLOCK)
        starts = [(s.ts - T0).total_seconds() for s in sigs]
        assert starts == pytest.approx([0.013 + k * 0.4 for k in range(len(starts))], abs=SEG_S)
        assert len(starts) == int((n / FS - 0.033) / 0.4) + 1

    def test_burst_longer_than_the_maximum_across_the_boundary_is_ignored(self):
        an = make_analyzer(signal_max_duration_ms=40)
        sigs = detect_blocks(an, burst_iq(2 * self.BLOCK, seg_s(self.NB - 30), 60 * SEG_S), self.BLOCK)
        assert sigs == []

    def test_burst_starting_in_the_first_segment_of_the_previous_block(self):
        # a short block, the burst covers all of the first one
        block = 16 * NPERSEG
        an = make_analyzer(analysis_block_samples=block, signal_max_duration_ms=50)
        sigs = detect_blocks(an, burst_iq(3 * block, seg_s(16), 20 * SEG_S), block)
        assert len(sigs) == 1
        assert (sigs[0].ts - T0).total_seconds() == pytest.approx(16 * SEG_S, abs=US)


class TestStreaming:
    """The callback, ring and consumer thread together, without an SDR."""

    @pytest.fixture
    def an(self):
        import threading

        an = make_analyzer(sdr_timeout_s=5)
        an.last_state = None
        an.analysis_block_samples = an._finalize_analysis_block_samples()
        an._ring_capacity = int(FS * an.sdr_timeout_s)
        an._ring_buffer = np.empty(an._ring_capacity, dtype=np.complex64)
        an._ring_cv = threading.Condition()
        an._consumer_stop = threading.Event()
        an.sdr = type("Sdr", (), {"cancelled": False, "cancel_read_async": lambda self: setattr(self, "cancelled", True)})()
        yield an
        signal.alarm(0)

    def signals(self, an):
        out = []
        while not an.signal_queue.empty():
            msg = an.signal_queue.get_nowait()
            if isinstance(msg, Signal):
                out.append(msg)
        return out

    def test_sample_clock_starts_one_buffer_before_the_first_callback(self, an):
        before = datetime.datetime.now()
        an._producer_enqueue(np.zeros(131072, dtype=np.complex64), None)
        after = datetime.datetime.now()
        offset = datetime.timedelta(seconds=131072 / FS)
        assert before - offset <= an._sample_clock_t0 <= after - offset

    def test_bursts_are_detected_across_callbacks_and_blocks(self, an):
        import threading

        n = 4 * an.analysis_block_samples
        iq = burst_iq(n, 0.29, 0.02) + burst_iq(n, 0.75, 0.025, tone_bin=-20, noise=0)
        consumer = threading.Thread(target=an._analysis_consumer_loop)
        consumer.start()
        # odd callback lengths, the blocks do not align with them
        for i in range(0, n, 131072):
            an._producer_enqueue(iq[i : i + 131072], None)
        an._consumer_stop.set()
        with an._ring_cv:
            an._ring_cv.notify_all()
        consumer.join(timeout=10)
        assert not consumer.is_alive()

        sigs = self.signals(an)
        assert [round((s.frequency - CENTER) / BIN_HZ) for s in sigs] == [TONE_BIN, -20]
        t0 = an._sample_clock_t0
        assert [(s.ts - t0.astimezone()).total_seconds() for s in sigs] == pytest.approx([0.29, 0.75], abs=SEG_S)
        # the bursts are not aligned to segments, the partially covered segments at both edges count for a strong one
        assert [s.duration.total_seconds() for s in sigs] == pytest.approx([0.02, 0.025], abs=2 * SEG_S)

    def test_ring_overflow_stops_streaming(self, an):
        an._producer_enqueue(np.zeros(an._ring_capacity - 10, dtype=np.complex64), None)
        assert not an.sdr.cancelled
        an._producer_enqueue(np.zeros(11, dtype=np.complex64), None)
        assert an.sdr.cancelled
        assert an._consumer_stop.is_set()
        assert an.last_state.state is StateMessage.State.STOPPED


