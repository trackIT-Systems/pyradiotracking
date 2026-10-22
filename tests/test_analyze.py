import datetime
import logging
import multiprocessing
import queue
import signal

import numpy as np
import pytest
import scipy.signal

from radiotracking import GainMessage, Signal, StateMessage, dB
from radiotracking.analyze import (
    R82XX_VGA_BASE_GAIN,
    SignalAnalyzer,
    resolve_block_samples,
    warn_if_block_too_short,
)

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
        spectrum_interval_s=0,
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


def iq_bytes(iq):
    """The bytes an RTL-SDR would deliver for IQ samples (interleaved I and Q, 127.5 is zero)."""
    raw = np.empty(2 * len(iq), dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    return raw


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

    @pytest.mark.parametrize(
        "field,getter,expected",
        [("lna_gain", "lna", "0.0 .. 33.5 dB"), ("mixer_gain", "mixer", "0.0 .. 16.1 dB"), ("vga_gain", "vga", "-4.7 .. 40.8 dB")],
    )
    def test_error_message_gives_the_measured_range(self, field, getter, expected):
        an = make_analyzer()
        setattr(an, field, -1)
        with pytest.raises(ValueError, match=expected.replace(".", r"\.")):
            getattr(an, f"_r82xx_get_{getter}_gain")()

    def test_vga_steps_are_not_the_datasheet_ones(self):
        # 3.5 dB per step from -12 dB would give 12.5 dB at index 7 and -5 dB at index 2
        an = make_analyzer(vga_gain=7)
        assert an._r82xx_get_vga_gain() + R82XX_VGA_BASE_GAIN / 10 == pytest.approx(14.9)
        assert make_analyzer(vga_gain=2)._r82xx_get_vga_gain() + R82XX_VGA_BASE_GAIN / 10 == pytest.approx(0.5)


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

    def test_stop_clears_a_pending_stall_timeout(self, monkeypatch):
        alarms = []
        monkeypatch.setattr(signal, "alarm", lambda s: alarms.append(s))
        make_analyzer().handle_signal(signal.SIGTERM, None)
        assert alarms == [0]

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

            def read_bytes_async(self, *args):
                self.streamed = True

        monkeypatch.setattr(rtlsdr.rtlsdr, "RtlSdr", FakeSdr)
        an.run()
        assert sdrs[0].closed
        assert not sdrs[0].streamed
        assert an.streaming_ts.value == 0.0

    def fake_streaming_sdr(self, monkeypatch, an, error, stop_first):
        import rtlsdr.rtlsdr

        class FakeSdr:
            closed = False

            def __init__(self, index):
                self.sample_rate = FS

            def set_manual_gain_enabled(self, enabled):
                pass

            def set_agc_mode(self, enabled):
                pass

            def close(self):
                FakeSdr.closed = True

            def cancel_read_async(self):
                pass

            def read_bytes_async(self, *args):
                if stop_first:
                    an.handle_signal(signal.SIGTERM, None)
                raise error

        monkeypatch.setattr(rtlsdr.rtlsdr, "RtlSdr", FakeSdr)
        # no librtlsdr in the test environment, the gain falls back to sdr.gain
        monkeypatch.setattr(rtlsdr, "librtlsdr", None, raising=False)
        return FakeSdr

    def test_usb_error_after_a_stop_request_ends_streaming_cleanly(self, monkeypatch):
        import rtlsdr.rtlsdr

        an = make_analyzer()
        sdr = self.fake_streaming_sdr(monkeypatch, an, rtlsdr.rtlsdr.LibUSBError("cancelled"), stop_first=True)
        an.run()
        assert sdr.closed

    def test_usb_error_while_streaming_is_raised(self, monkeypatch):
        import rtlsdr.rtlsdr

        an = make_analyzer()
        sdr = self.fake_streaming_sdr(monkeypatch, an, rtlsdr.rtlsdr.LibUSBError("gone"), stop_first=False)
        with pytest.raises(rtlsdr.rtlsdr.LibUSBError):
            an.run()
        assert sdr.closed


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

    def test_simultaneous_tags_on_distant_frequencies_are_both_detected(self):
        # two tags 85 kHz apart, the weak one starts 5 ms after the strong one; the probability of a pulse overlapping
        # some other tag's pulse grows with the number of tags in range (~4 % per tag at 20 ms pulses every second)
        an = make_analyzer(signal_threshold_dbw=-120)
        block = an._finalize_analysis_block_samples()
        iq = burst_iq(block, 0.1, 0.02, tone_bin=TONE_BIN, amp=0.1) + burst_iq(block, 0.105, 0.02, tone_bin=-30, amp=1e-3, noise=0)
        sigs, _ = detect(an, iq)
        assert sorted(round((s.frequency - CENTER) / BIN_HZ) for s in sigs) == [-30, TONE_BIN]

    def test_distant_signal_is_no_shadow(self, make_signal):
        # two transmitters sending at the same time are both kept, see #6
        loud = make_signal(frequency=150_000_000, max_dbw=-40)
        weak = make_signal(frequency=150_100_000, max_dbw=-60)
        assert make_analyzer().filter_shadow_signals([loud, weak]) == [loud, weak]

    def test_shadow_bins_limit(self, make_signal):
        bin_hz = FS / NPERSEG
        loud = make_signal(frequency=150_000_000, max_dbw=-40)
        near = make_signal(frequency=150_000_000 + 4 * bin_hz, max_dbw=-60)
        far = make_signal(frequency=150_000_000 + 5 * bin_hz, max_dbw=-60)
        assert make_analyzer().filter_shadow_signals([loud, near, far]) == [loud, far]
        assert make_analyzer(shadow_bins=5).filter_shadow_signals([loud, near, far]) == [loud]
        assert make_analyzer(shadow_bins=0).filter_shadow_signals([loud, near, far]) == [loud, near, far]


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


@pytest.mark.parametrize(
    "configured,expected",
    [(None, 131072), (131072, 131072), (65536, 65536), (256, 256), (1, 256), (300000, 300032), (1000, 1024)],
)
def test_resolve_callback_length(configured, expected):
    from radiotracking.analyze import resolve_callback_length

    assert resolve_callback_length(configured) == expected


def test_callback_length_that_librtlsdr_would_ignore_warns(caplog):
    from radiotracking.analyze import resolve_callback_length

    with caplog.at_level(logging.WARNING):
        resolve_callback_length(131072)
    assert caplog.text == ""
    with caplog.at_level(logging.WARNING):
        resolve_callback_length(300000)
    assert "not a multiple of 256" in caplog.text


@pytest.mark.parametrize("configured", [0, -256])
def test_callback_length_must_be_positive(configured):
    from radiotracking.analyze import resolve_callback_length

    with pytest.raises(ValueError):
        resolve_callback_length(configured)


@pytest.mark.parametrize(
    "timeout,callback,block,expected",
    [(2, 131072, 90112, 600000), (1, 300032, 90112, 390144), (2, 1_000_000, 90112, 1_090_112)],
)
def test_ring_holds_a_callback_and_a_block(timeout, callback, block, expected):
    from radiotracking.analyze import resolve_ring_capacity

    assert resolve_ring_capacity(FS, timeout, callback, block) == expected


class TestStreaming:
    """The callback, ring and consumer thread together, without an SDR."""

    @pytest.fixture
    def an(self):
        import threading

        # hamming leaks further than shadow_bins from a strong burst, see #6
        an = make_analyzer(sdr_timeout_s=5, fft_window="blackmanharris")
        an.last_state = None
        an.analysis_block_samples = an._finalize_analysis_block_samples()
        an._ring_capacity = int(FS * an.sdr_timeout_s)
        an._ring_buffer = np.empty((an._ring_capacity, 2), dtype=np.uint8)
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
        an._producer_enqueue(np.zeros(2 * 131072, dtype=np.uint8), None)
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
            an._producer_enqueue(iq_bytes(iq[i : i + 131072]), None)
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

    def test_spectrum_is_published_by_the_consumer_loop(self, an):
        import threading

        from radiotracking import SpectrumMessage

        an.spectrum_interval_s = 2 * an.analysis_block_samples / FS
        consumer = threading.Thread(target=an._analysis_consumer_loop)
        consumer.start()
        iq = burst_iq(4 * an.analysis_block_samples, 0.0, 0.0)
        for i in range(0, len(iq), 131072):
            an._producer_enqueue(iq_bytes(iq[i : i + 131072]), None)
        an._consumer_stop.set()
        with an._ring_cv:
            an._ring_cv.notify_all()
        consumer.join(timeout=10)
        msgs = []
        while not an.signal_queue.empty():
            msg = an.signal_queue.get_nowait()
            if isinstance(msg, SpectrumMessage):
                msgs.append(msg)
        # the first block of the stream is left out (start transient), the fourth starts an incomplete interval
        assert len(msgs) == 1
        block = datetime.timedelta(seconds=an.analysis_block_samples / FS)
        assert (msgs[0].ts - an._sample_clock_t0.astimezone()).total_seconds() == pytest.approx(block.total_seconds(), abs=US)
        assert msgs[0].duration.total_seconds() == pytest.approx(2 * an.analysis_block_samples / FS, abs=US)

    def test_ring_overflow_stops_streaming(self, an):
        an._producer_enqueue(np.zeros(2 * (an._ring_capacity - 10), dtype=np.uint8), None)
        assert not an.sdr.cancelled
        an._producer_enqueue(np.zeros(2 * 11, dtype=np.uint8), None)
        assert an.sdr.cancelled
        assert an._consumer_stop.is_set()
        assert an.last_state.state is StateMessage.State.STOPPED


@pytest.mark.parametrize(
    "timeout,callback,expected",
    [(2, 131072, 2), (1, 131072, 1), (1, 300032, 2), (2, 300032, 2), (1, 600064, 4), (5, 300032, 5), (1, 256, 1)],
)
def test_sdr_timeout_covers_a_callback(timeout, callback, expected):
    from radiotracking.analyze import resolve_sdr_timeout

    assert resolve_sdr_timeout(timeout, callback, FS) == expected


def test_analyzer_raises_a_too_short_timeout(caplog):
    with caplog.at_level(logging.WARNING):
        an = make_analyzer(sdr_timeout_s=1, sdr_callback_length=300000)
    assert an.sdr_timeout_s == 2
    assert "sdr_timeout_s (1 s) is too short" in caplog.text


class TestSpectrum:
    BLOCK = 30208  # 0.1 s

    def run(self, an, iq):
        """Analyse consecutive blocks like the consumer loop does, return the spectrum messages."""
        from radiotracking import SpectrumMessage
        from radiotracking.analyze import noise_levels

        for i in range(len(iq) // self.BLOCK):
            freqs, times, spec = spectrogram(an, iq[i * self.BLOCK : (i + 1) * self.BLOCK])
            noise = noise_levels(spec)
            ts = T0 + datetime.timedelta(seconds=i * self.BLOCK / FS)
            an.extract_signals(freqs, times, spec, ts, noise)
            an._accumulate_spectrum(freqs, spec, noise, ts, self.BLOCK)
            an._spectrogram_last = spec
        out = []
        while not an.signal_queue.empty():
            msg = an.signal_queue.get_nowait()
            if isinstance(msg, SpectrumMessage):
                out.append(msg)
        return out

    def analyzer(self, **over):
        # an interval of three blocks
        return make_analyzer(spectrum_interval_s=3 * self.BLOCK / FS, **over)

    def test_noise_levels_of_complex_noise(self):
        from radiotracking.analyze import noise_levels

        an = make_analyzer()
        _, _, spec = spectrogram(an, burst_iq(90112, 0.0, 0.0))
        assert dB(np.median(noise_levels(spec))) == pytest.approx(dB(2e-6 / FS), abs=0.5)

    def test_precomputed_noise_gives_the_same_signals(self):
        from radiotracking.analyze import noise_levels

        an = make_analyzer()
        freqs, times, spec = spectrogram(an, burst_iq(30208, 0.03, 0.02) + burst_iq(30208, 0.06, 0.02, tone_bin=-50, noise=0))
        assert repr(an.extract_signals(freqs, times, spec, T0)) == repr(
            an.extract_signals(freqs, times, spec, T0, noise_levels(spec))
        )

    def test_one_message_per_complete_interval(self):
        an = self.analyzer()
        assert self.run(an, burst_iq(2 * self.BLOCK, 0.0, 0.0)) == []
        an = self.analyzer()
        msgs = self.run(an, burst_iq(7 * self.BLOCK, 0.0, 0.0))
        # the seventh block starts an interval that is not complete
        assert len(msgs) == 2
        assert [m.ts for m in msgs] == [T0, T0 + datetime.timedelta(seconds=3 * self.BLOCK / FS)]
        assert all(m.ts.tzinfo is not None for m in msgs)
        assert all(m.duration.total_seconds() == pytest.approx(3 * self.BLOCK / FS, abs=US) for m in msgs)

    def test_bins_ascend_over_the_band(self):
        (msg,) = self.run(self.analyzer(), burst_iq(3 * self.BLOCK, 0.0, 0.0))
        assert len(msg.noise) == len(msg.snr) == NPERSEG
        assert msg.bin_width == BIN_HZ
        assert msg.frequency == pytest.approx(CENTER - FS / 2)
        assert msg.frequencies[NPERSEG // 2] == pytest.approx(CENTER)

    @pytest.mark.parametrize("tone_bin", [TONE_BIN, -30])
    def test_a_burst_is_the_snr_maximum_in_its_bin(self, tone_bin):
        iq = burst_iq(3 * self.BLOCK, 0.13, 0.02, tone_bin=tone_bin, amp=4e-3)
        an = self.analyzer(signal_threshold_dbw=-120)
        (msg,) = self.run(an, iq)
        peak = int(np.argmax(msg.snr))
        assert msg.frequencies[peak] == pytest.approx(CENTER + tone_bin * BIN_HZ)

        # the burst's SNR as reported by its signal, the peak of a constant tone equals its average
        (sig,) = detect(make_analyzer(signal_threshold_dbw=-120), iq[self.BLOCK : 2 * self.BLOCK])[0]
        assert msg.snr[peak] == pytest.approx(sig.snr, abs=1.5)
        # noise only: the maximum of ~350 exponentially distributed powers is ~8 dB above their mean
        quiet = [snr for i, snr in enumerate(msg.snr) if abs(i - peak) > 2]
        assert max(quiet) < 15

    def test_noise_is_not_raised_by_a_burst(self):
        # the median over the blocks of an interval: with only a few blocks, a burst in one of them moves it to a
        # neighbouring block's estimate, a 300 s interval has ~1000 blocks
        def interval():
            return make_analyzer(spectrum_interval_s=9 * self.BLOCK / FS)

        quiet = self.run(interval(), burst_iq(9 * self.BLOCK, 0.0, 0.0))[0]
        loud = self.run(interval(), burst_iq(9 * self.BLOCK, 0.43, 0.02, amp=1.0))[0]
        assert np.array(loud.noise) == pytest.approx(np.array(quiet.noise), abs=0.5)
        assert max(loud.snr) > 70

    def test_calibration_shifts_the_noise_but_not_the_snr(self):
        iq = burst_iq(3 * self.BLOCK, 0.13, 0.02)
        plain = self.run(self.analyzer(), iq)[0]
        calibrated = self.run(self.analyzer(calibration_db=6.0), iq)[0]
        assert np.array(plain.noise) - np.array(calibrated.noise) == pytest.approx(6.0)
        assert calibrated.snr == pytest.approx(plain.snr)

    def test_disabled(self):
        assert self.run(make_analyzer(spectrum_interval_s=0), burst_iq(6 * self.BLOCK, 0.0, 0.0)) == []


class TestBytes:
    def test_iq_from_bytes_equals_pyrtlsdr(self):
        """The table gives exactly the values of pyrtlsdr's packed_bytes_to_iq, stored as complex64."""
        from radiotracking.analyze import iq_from_bytes

        raw = np.array([(i, q) for i in range(256) for q in range(256)], dtype=np.uint8).ravel()
        # pyrtlsdr's packed_bytes_to_iq
        expected = raw.astype(np.float64).view(np.complex128)
        expected /= 127.5
        expected -= 1 + 1j
        expected = expected.astype(np.complex64)
        assert np.array_equal(iq_from_bytes(raw).view(np.uint64), expected.view(np.uint64))
        assert np.array_equal(iq_from_bytes(raw.reshape(-1, 2)), iq_from_bytes(raw))

    def test_ring_of_byte_rows_wraps_around(self):
        an = make_analyzer()
        an._ring_capacity = 4
        an._ring_buffer = np.empty((4, 2), dtype=np.uint8)
        rows = np.arange(12, dtype=np.uint8).reshape(-1, 2)
        assert an._ring_push(rows[:3])
        assert an._ring_pop(2).tolist() == [[0, 1], [2, 3]]
        assert an._ring_push(rows[3:6])  # wraps
        assert an._ring_pop(4).tolist() == [[4, 5], [6, 7], [8, 9], [10, 11]]

    def test_callback_takes_the_ctypes_buffer_of_librtlsdr(self):
        import ctypes
        import threading

        an = make_analyzer(sdr_timeout_s=5)
        an.last_state = None
        an._ring_capacity = 1024
        an._ring_buffer = np.empty((1024, 2), dtype=np.uint8)
        an._ring_cv = threading.Condition()
        an._consumer_stop = threading.Event()
        data = bytes(range(256)) * 2
        try:
            an._producer_enqueue((ctypes.c_ubyte * len(data)).from_buffer_copy(data), None)
        finally:
            signal.alarm(0)
        assert an._ring_count == 256
        assert an._ring_pop(2).tolist() == [[0, 1], [2, 3]]


class TestExactShortcuts:
    """Faster forms of scipy's spectrogram and np.median, which must give identical values."""

    @pytest.mark.parametrize("window", ["hamming", "hann", "blackmanharris", ("kaiser", 8.0), "boxcar"])
    @pytest.mark.parametrize("sample_rate,nperseg", [(FS, 256), (1_200_000, 1024), (250_000, 100), (FS, 1)])
    def test_spectrogram_equals_scipy(self, window, sample_rate, nperseg):
        from radiotracking.analyze import spectrogram_of, spectrogram_window

        rng = np.random.default_rng(3)
        n = 40 * nperseg
        iq = (1e-2 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)
        iq += (0.5 * np.exp(2j * np.pi * 0.2 * np.arange(n))).astype(np.complex64)
        expected = scipy.signal.spectrogram(
            iq, fs=sample_rate, window=window, nperseg=nperseg, noverlap=0, return_onesided=False
        )
        actual = spectrogram_of(iq, sample_rate, *spectrogram_window(window, nperseg, sample_rate))
        for e, a in zip(expected, actual):
            assert a.dtype == e.dtype
            assert np.array_equal(a, e)

    @pytest.mark.parametrize("segments", [352, 351, 1, 2])
    def test_noise_levels_equal_the_median(self, segments):
        from radiotracking.analyze import LN2, noise_levels

        rng = np.random.default_rng(segments)
        spec = rng.exponential(1e-10, size=(segments, 64)).astype(np.float32).T  # scipy's memory order
        assert np.array_equal(noise_levels(spec), np.median(spec, axis=1) / LN2)

    def test_window_follows_the_sample_rate(self):
        an = make_analyzer(sample_rate=250_000)
        an._analyze_block(burst_iq(30208, 0.0, 0.0, fs=250_000), T0)
        first = an._window
        an.sample_rate = FS
        an._analyze_block(burst_iq(30208, 0.0, 0.0), T0)
        assert an._window_rate == FS
        assert an._window[1] != first[1]


class TestSampleClock:
    """The sample clock is anchored to the wall clock, so signal timestamps don't drift with the SDR's crystal."""

    CALLBACK = 131072

    def run(self, an, seconds, ppm=0.0, latency_ms=(0.0, 5.0), seed=0, step_at=None, step_s=0.0, first_latency_ms=None):
        """Simulate callbacks of an SDR whose crystal is off by `ppm`, return the timestamp error (s) at the end."""
        rng = np.random.default_rng(seed)
        t_wall0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        received = 0
        for k in range(int(seconds * FS / self.CALLBACK)):
            received += self.CALLBACK
            # the true wall time of the last sample: the SDR's clock runs ppm too fast or slow
            true_end = received / FS / (1 + ppm * 1e-6)
            wall = true_end + rng.uniform(*latency_ms) / 1000
            if k == 0 and first_latency_ms is not None:
                wall = true_end + first_latency_ms / 1000
            if step_at is not None and true_end >= step_at:
                wall += step_s
            an._anchor_sample_clock(t_wall0 + datetime.timedelta(seconds=wall), self.CALLBACK)
        # timestamp of the last sample as a signal would get it, against its true wall time
        stamped = an._sample_clock_t0 + datetime.timedelta(seconds=received / FS)
        true = t_wall0 + datetime.timedelta(seconds=true_end + (step_s if step_at is not None else 0))
        return (stamped - true).total_seconds()

    def test_first_callback_anchors_one_buffer_before(self):
        an = make_analyzer()
        now = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        an._anchor_sample_clock(now, self.CALLBACK)
        assert an._sample_clock_t0 == now - datetime.timedelta(seconds=self.CALLBACK / FS)

    @pytest.mark.parametrize("ppm", [0.4, 50, -50])
    def test_crystal_drift_is_corrected(self, ppm):
        # one hour: without anchoring, 50 ppm are 180 ms
        error = self.run(make_analyzer(), 3600, ppm=ppm)
        assert abs(error) < 0.002

    def test_latency_of_the_first_callback_is_corrected(self):
        # the SDRs of a station started 1-9 ms apart, depending on the latency of their first callback
        assert abs(self.run(make_analyzer(), 30, latency_ms=(0.0, 0.2), first_latency_ms=9.0)) < 0.0005

    def test_wall_clock_steps_are_followed(self):
        # e.g. NTP correcting the clock by a second after 15 s
        error = self.run(make_analyzer(), 60, step_at=15, step_s=1.0)
        assert abs(error) < 0.01


class TestZeroNoise:
    """An SDR delivering constant samples has frequencies without noise, which must not give infinite values."""

    def spectrogram_with_silent_bin(self):
        spec = np.full((8, 40), 1e-12, dtype=np.float32)
        spec[3] = 0.0
        spec[3, 10:20] = 1e-6  # a signal in the silent frequency
        return spec

    def test_noise_level_is_finite(self):
        from radiotracking.analyze import noise_levels

        noise = noise_levels(self.spectrogram_with_silent_bin())
        assert np.all(noise > 0)
        assert np.all(np.isfinite(dB(noise)))

    def test_signal_in_a_silent_frequency_has_finite_values(self):
        import json

        from radiotracking.consume import jsonify

        an = make_analyzer(signal_threshold_dbw=-130, signal_min_duration_ms=0.1)
        freqs = np.fft.fftfreq(8, 1 / FS)
        times = (np.arange(40) + 0.5) * NPERSEG / FS
        (sig,) = an.extract_signals(freqs, times, self.spectrogram_with_silent_bin(), T0)
        assert np.isfinite(sig.snr) and np.isfinite(sig.noise)
        json.dumps(sig.as_dict, default=jsonify, allow_nan=False)

    def test_spectrum_of_a_silent_frequency_has_finite_values(self):
        an = make_analyzer(spectrum_interval_s=1e-9)
        spec = self.spectrogram_with_silent_bin()
        from radiotracking.analyze import noise_levels

        an._accumulate_spectrum(np.fft.fftfreq(8, 1 / FS), spec, noise_levels(spec), T0, 40 * NPERSEG)
        msg = an.signal_queue.get_nowait()
        assert all(np.isfinite(v) for v in msg.noise + msg.snr)


class TestCalibratedPowers:
    """Signals of a device with ``fullscale_dbm`` and a known gain carry their power in dBm at the antenna port."""

    GAIN = 64.5
    FULLSCALE = -20.0

    def burst(self, an, amp=0.1):
        sigs, _ = detect(an, burst_iq(90112, 0.1, 0.02, amp=amp, noise=1e-5))
        assert len(sigs) == 1
        return sigs[0]

    def test_tone_power_is_the_amplitude_referred_to_the_antenna(self):
        an = make_analyzer(fullscale_dbm=self.FULLSCALE)
        an.set_gain_reference(self.GAIN)
        sig = self.burst(an, 0.1)
        # a tone of amplitude 0.1 has the power -20 dB re full scale, at the antenna port: - gain + full scale reference
        assert sig.max_dBm == pytest.approx(-20.0 - self.GAIN + self.FULLSCALE, abs=0.1)
        assert sig.avg_dBm == pytest.approx(sig.max_dBm, abs=0.2)

    def test_snr_is_the_difference_of_max_and_noise(self):
        an = make_analyzer(fullscale_dbm=self.FULLSCALE)
        an.set_gain_reference(self.GAIN)
        sig = self.burst(an)
        assert sig.max_dBm - sig.noise_dBm == pytest.approx(sig.max - sig.noise, abs=1e-4)

    def test_dbw_values_do_not_change(self):
        plain = make_analyzer()
        cal = make_analyzer(fullscale_dbm=self.FULLSCALE)
        cal.set_gain_reference(self.GAIN)
        a, b = self.burst(plain), self.burst(cal)
        assert a.as_list == b.as_list[:9]
        assert a.max_dBm is None and a.header == Signal.header

    def test_offset_converts_the_published_dbw(self):
        an = make_analyzer(fullscale_dbm=self.FULLSCALE, calibration_db=3.0)
        offset = an.set_gain_reference(self.GAIN)
        sig = self.burst(an)
        assert sig.max + offset == pytest.approx(sig.max_dBm, abs=1e-4)
        assert sig.noise + offset == pytest.approx(sig.noise_dBm, abs=1e-4)

    def test_tone_offset_follows_the_window_and_the_bins(self):
        assert make_analyzer().tone_offset_db() == pytest.approx(32.03, abs=0.01)
        # the bin width is the same at 1.2 MHz with 1024 bins
        assert make_analyzer(sample_rate=1_200_000, fft_nperseg=1024).tone_offset_db() == pytest.approx(32.03, abs=0.01)

    def test_without_reference_or_gain_nothing_is_calibrated(self):
        assert make_analyzer().set_gain_reference(self.GAIN) is None
        an = make_analyzer(fullscale_dbm=self.FULLSCALE)
        assert an.set_gain_reference(None) is None
        assert self.burst(an).max_dBm is None

    def test_gain_message(self):
        an = make_analyzer(fullscale_dbm=self.FULLSCALE, calibration_db=1.0)
        an.set_gain_reference(an._r82xx_get_gain())
        msg = an._gain_message(T0)
        assert msg.gain_dB == pytest.approx(64.5)
        assert (msg.lna, msg.mixer, msg.vga) == (15, 14, 7)
        assert msg.offset_dB == pytest.approx(1.0 + 32.03 - 64.5 + self.FULLSCALE, abs=0.01)

        plain = make_analyzer()._gain_message(T0)
        assert plain.gain_dB is None and plain.offset_dB is None

    def test_total_gain_of_the_stages(self):
        gains = {(15, 14, 15): 90.4, (15, 14, 3): 53.1, (15, 14, 7): 64.5, (15, 14, 11): 76.1}
        for (lna, mixer, vga), total in gains.items():
            assert make_analyzer(lna_gain=lna, mixer_gain=mixer, vga_gain=vga)._r82xx_get_gain() == pytest.approx(total)


class TestGainSetup:
    """The gain stages are set with rtlsdr_set_tuner_gain_ext, other cases are reported as a fallback."""

    @pytest.fixture
    def setup(self, monkeypatch):
        import types

        import rtlsdr.rtlsdr

        calls = []

        class FakeSdr:
            dev_p = object()
            gain = 49.6

            def __init__(self, index):
                self.sample_rate = FS

            def set_manual_gain_enabled(self, enabled):
                pass

            def set_agc_mode(self, enabled):
                pass

            def close(self):
                pass

            def cancel_read_async(self):
                pass

            def read_bytes_async(self, *args):
                raise rtlsdr.rtlsdr.LibUSBError("end")

        def configure(ret=0, expect=rtlsdr.rtlsdr.LibUSBError, **over):
            monkeypatch.setattr(rtlsdr.rtlsdr, "RtlSdr", FakeSdr)
            monkeypatch.setattr(
                rtlsdr,
                "librtlsdr",
                types.SimpleNamespace(rtlsdr_set_tuner_gain_ext=lambda *args: calls.append(args) or ret),
                raising=False,
            )
            an = make_analyzer(**over)
            with pytest.raises(expect) as exc:
                an.run()
            configure.error = exc.value
            return an

        configure.calls = calls
        return configure

    def test_stages_are_set_and_logged(self, setup, caplog):
        with caplog.at_level(logging.INFO):
            an = setup(vga_gain=7)
        assert setup.calls[0][1:] == (15, 14, 7)
        assert an.total_gain_db == pytest.approx(64.5)
        assert "LNA index 15 (33.5 dB) + mixer index 14 (16.1 dB) + VGA index 7 (14.9 dB) = 64.5 dB" in caplog.text

    def test_gain_message_is_queued(self, setup):
        an = setup(vga_gain=3)
        msg = an.signal_queue.get_nowait()
        assert isinstance(msg, GainMessage)
        assert msg.gain_dB == pytest.approx(53.1)

    def test_failed_call_falls_back_explicitly(self, setup, caplog):
        with caplog.at_level(logging.WARNING):
            an = setup(ret=-1)
        assert an.total_gain_db is None
        assert "the gain stages can not be set separately" in caplog.text
        assert "set the gain to 49.6 dB" in caplog.text
        assert an.signal_queue.get_nowait().gain_dB is None

    def test_invalid_index_does_not_fall_back(self, setup):
        setup(expect=ValueError, vga_gain=16)
        assert "vga_gain index 16 not in 0 .. 15" in str(setup.error)
        assert setup.calls == []
