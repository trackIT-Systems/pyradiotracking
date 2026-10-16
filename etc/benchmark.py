#!/usr/bin/env python3
"""
Time the analysis of radiotracking per block, on synthetic IQ samples as an RTL-SDR delivers them.

Three cases: a quiet band, a tag ~50 dB above the noise, and a close tag near full scale, whose spectral leakage
reaches every frequency. The tags send 25 ms pulses every second, between two FFT bins (the worst case for leakage).

    python3 etc/benchmark.py --sample-rate 1200000 --fft-nperseg 1024
"""

import argparse
import datetime
import multiprocessing
import queue
import statistics
import sys
import time
import types

import numpy as np

try:
    import rtlsdr  # noqa: F401
except ImportError:
    # without librtlsdr (e.g. on a development machine), radiotracking only needs the names
    pkg = types.ModuleType("rtlsdr")
    pkg.rtlsdr = types.ModuleType("rtlsdr.rtlsdr")
    pkg.rtlsdr.RtlSdr = object
    pkg.rtlsdr.LibUSBError = Exception
    pkg.librtlsdr = types.ModuleType("rtlsdr.librtlsdr")
    sys.modules.update({"rtlsdr": pkg, "rtlsdr.rtlsdr": pkg.rtlsdr, "rtlsdr.librtlsdr": pkg.librtlsdr})

from radiotracking.analyze import SignalAnalyzer, iq_from_bytes  # noqa: E402

parser = argparse.ArgumentParser(
    prog="benchmark",
    description="Time the analysis of radiotracking per block",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-s", "--sample-rate", default=300_000, type=int, help="sample rate (Hz)")
parser.add_argument("-n", "--fft-nperseg", default=256, type=int, help="fft number of samples")
parser.add_argument("-t", "--signal-threshold-dbw", default=-95.0, type=float, help="signal threshold (dBW)")
parser.add_argument("-r", "--snr-threshold-db", default=3.0, type=float, help="SNR threshold (dB)")
parser.add_argument("--seconds", default=6.0, type=float, help="seconds of IQ samples per case")
parser.add_argument("--noise-counts", default=0.6, type=float, help="noise level (standard deviation, ADC counts)")

CASES = {"quiet band": 0.0, "tag ~50 dB": 0.08, "close tag": 0.9}
"""Tag amplitude relative to the ADC's full scale."""


def raw_samples(sample_rate: int, nperseg: int, seconds: float, amplitude: float, noise_counts: float) -> np.ndarray:
    """uint8 IQ bytes as the SDR delivers them, with 25 ms pulses every second half way between two bins."""
    rng = np.random.default_rng(0)
    n = int(seconds * sample_rate)
    iq = noise_counts / 127.5 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    frequency = (nperseg // 5 + 0.5) * sample_rate / nperseg
    for start in np.arange(0.1, seconds, 1.0):
        a, b = int(start * sample_rate), min(n, int((start + 0.025) * sample_rate))
        iq[a:b] += amplitude * np.exp(2j * np.pi * frequency * np.arange(a, b) / sample_rate)
    raw = np.empty(2 * n, dtype=np.uint8)
    raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
    return raw


def analyzer(args) -> SignalAnalyzer:
    return SignalAnalyzer(
        device="0",
        calibration_db=0.0,
        sample_rate=args.sample_rate,
        center_freq=150_150_000,
        gain=49.6,
        lna_gain=15,
        mixer_gain=14,
        vga_gain=7,
        tuner_bandwidth=0,
        fft_nperseg=args.fft_nperseg,
        fft_window="hamming",
        signal_min_duration_ms=8,
        signal_max_duration_ms=50,
        signal_threshold_dbw=args.signal_threshold_dbw,
        snr_threshold_db=args.snr_threshold_db,
        verbose=0,
        sdr_max_restart=0,
        sdr_timeout_s=2,
        state_update_s=300,
        sdr_callback_length=None,
        analysis_block_samples=None,
        signal_queue=queue.Queue(),
        last_data_ts=multiprocessing.Value("d", 0.0),
        streaming_ts=multiprocessing.Value("d", 0.0),
        spectrum_interval_s=300,
    )


def main():
    args = parser.parse_args()
    block = analyzer(args)._finalize_analysis_block_samples()
    block_s = block / args.sample_rate
    print(
        f"{args.sample_rate / 1e6:g} MHz, {args.fft_nperseg} bins, blocks of {block} samples ({block_s * 1000:.0f} ms), "
        f"medians per block in ms"
    )

    for case, amplitude in CASES.items():
        raw = raw_samples(args.sample_rate, args.fft_nperseg, args.seconds, amplitude, args.noise_counts)
        an = analyzer(args)
        stages: dict[str, list[float]] = {"convert": []}
        published = raw_signals = 0
        for i in range(len(raw) // (2 * block)):
            chunk_raw = raw[2 * i * block : 2 * (i + 1) * block]
            start = time.perf_counter()
            chunk = iq_from_bytes(chunk_raw)
            stages["convert"].append(time.perf_counter() - start)

            ts = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(seconds=i * block_s)
            n_kept, n_raw, timings = an._analyze_block(chunk, ts)
            published += n_kept
            raw_signals += n_raw
            for stage, seconds in timings.items():
                stages.setdefault(stage, []).append(seconds)

        medians = {stage: statistics.median(values) * 1000 for stage, values in stages.items()}
        worst = max(sum(values[i] for values in stages.values()) for i in range(len(stages["convert"]))) * 1000
        total = sum(medians.values())
        print(
            f"{case:11s} total {total:6.1f} (worst {worst:6.1f}, {total / (block_s * 1000) * 100:4.1f} % of real time) | "
            + " ".join(f"{stage} {ms:5.1f}" for stage, ms in medians.items())
            + f" | {raw_signals} signals, {published} published"
        )


if __name__ == "__main__":
    main()
