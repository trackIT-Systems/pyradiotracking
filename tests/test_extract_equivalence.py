"""The vectorized signal extraction returns exactly the signals of the former per frequency search."""

import datetime

import numpy as np
import pytest

from radiotracking import Signal

from .conftest import T0
from .reference_extract import reference_extract_signals
from .test_analyze import BIN_HZ, CENTER, FS, NPERSEG, burst_iq, make_analyzer, spectrogram


def random_spectrograms(seed, blocks=4, bins=64, segments=40):
    """Exponentially distributed noise power with bursts of random frequency, start, length and strength."""
    rng = np.random.default_rng(seed)
    specs = rng.exponential(1e-10, size=(blocks, bins, segments)).astype(np.float32)
    full = specs.transpose(1, 0, 2).reshape(bins, blocks * segments)
    for _ in range(rng.integers(5, 40)):
        f = rng.integers(bins)
        a = rng.integers(blocks * segments)
        b = min(blocks * segments, a + rng.integers(1, 3 * segments // 2))
        full[f, a:b] *= np.float32(10 ** rng.uniform(0, 6))
        # leakage into the neighbours, with the same timing as the burst
        for d in (-1, 1):
            if 0 <= f + d < bins:
                full[f + d, a:b] *= np.float32(10 ** rng.uniform(0, 3))
    return [full[:, i * segments : (i + 1) * segments].copy() for i in range(blocks)]


def compare(an, specs, freqs, times):
    """Run consecutive blocks through the vectorized and the reference extraction, return the number of signals."""
    count = 0
    for i, spec in enumerate(specs):
        ts = T0 + datetime.timedelta(seconds=i * len(times) * an.fft_nperseg / an.sample_rate)
        expected = reference_extract_signals(an, freqs, times, spec, ts)
        assert repr(an.extract_signals(freqs, times, spec, ts)) == repr(expected), f"block {i}"

        kept, n_raw = an.detect_signals(freqs, times, spec, ts)
        assert n_raw == len(expected)
        assert repr(kept) == repr(an.filter_shadow_signals(expected)), f"block {i}"

        an._spectrogram_last = spec
        count += len(expected)
    return count


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize(
    "params",
    [
        dict(signal_threshold_dbw=-100, snr_threshold_db=5),
        dict(signal_threshold_dbw=-110, snr_threshold_db=3, signal_min_duration_ms=0.1, signal_max_duration_ms=50),
        dict(signal_threshold_dbw=-90, snr_threshold_db=10, signal_min_duration_ms=4, signal_max_duration_ms=20),
    ],
    ids=["default", "short-signals", "strict"],
)
def test_random_spectrograms(seed, params):
    an = make_analyzer(**params)
    specs = random_spectrograms(seed)
    freqs = np.fft.fftfreq(specs[0].shape[0], 1 / FS)
    times = (np.arange(specs[0].shape[1]) + 0.5) * NPERSEG / FS
    compare(an, specs, freqs, times)


def test_random_spectrograms_find_signals():
    """The random spectrograms are not trivial: signals start in previous blocks and touch block boundaries."""
    an = make_analyzer(signal_threshold_dbw=-110, snr_threshold_db=3, signal_min_duration_ms=0.1, signal_max_duration_ms=50)
    specs = random_spectrograms(0)
    freqs = np.fft.fftfreq(specs[0].shape[0], 1 / FS)
    times = (np.arange(specs[0].shape[1]) + 0.5) * NPERSEG / FS
    assert compare(an, specs, freqs, times) > 50


@pytest.mark.parametrize("amp", [4e-4, 1e-2, 1.0])
def test_bursts_across_blocks(amp):
    an = make_analyzer(signal_threshold_dbw=-120)
    block = 30208
    iq = burst_iq(4 * block, block / FS - 0.01, 0.02, amp=amp) + burst_iq(
        4 * block, 2 * block / FS - 24 * NPERSEG / FS, 24 * NPERSEG / FS, tone_bin=-30, amp=amp, noise=0
    )
    specs, freqs, times = [], None, None
    for i in range(4):
        freqs, times, spec = spectrogram(an, iq[i * block : (i + 1) * block])
        specs.append(spec)
    assert compare(an, specs, freqs, times) >= 2


def test_strong_leakage_is_filtered_before_signals_are_created():
    """A tone between two bins leaks into many frequencies, only the strongest signal is created."""
    an = make_analyzer(signal_threshold_dbw=-130)
    freqs, times, spec = spectrogram(an, burst_iq(90112, 0.1, 0.02, amp=1.0, tone_bin=43.5))
    kept, n_raw = an.detect_signals(freqs, times, spec, T0)
    assert n_raw > 50
    assert len(kept) == 1
    assert isinstance(kept[0], Signal)
    assert kept[0].frequency == pytest.approx(CENTER + 43 * BIN_HZ, abs=BIN_HZ)


def test_touching_signals_follow_the_rounded_timestamps():
    """
    Whether a signal starting right where a louder one ends is its shadow depends on the microsecond rounding of the
    timestamps and durations, exactly as in is_shadow_of: here the first ends at 17.066 ms, the second starts at 17.067.
    """
    an = make_analyzer(signal_threshold_dbw=-130, signal_min_duration_ms=0.1)
    spec = np.full((8, 40), 1e-12, dtype=np.float32)
    spec[2, 10:20] = 1e-6
    spec[5, 20:30] = 1e-8
    freqs = np.fft.fftfreq(8, 1 / FS)
    times = (np.arange(40) + 0.5) * NPERSEG / FS
    expected = an.filter_shadow_signals(an.extract_signals(freqs, times, spec, T0))
    kept, n_raw = an.detect_signals(freqs, times, spec, T0)
    assert n_raw == 2
    assert repr(kept) == repr(expected)
    assert len(kept) == 2


def test_empty_block():
    an = make_analyzer()
    assert an.detect_signals(np.zeros(4), np.zeros(0), np.zeros((4, 0), dtype=np.float32), T0) == ([], 0)
