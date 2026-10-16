# pyradiotracking

[![Latest version](https://img.shields.io/github/v/tag/trackIT-Systems/pyradiotracking?sort=semver&label=version)](https://github.com/trackIT-Systems/pyradiotracking/tags)
[![Tests](https://github.com/trackIT-Systems/pyradiotracking/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/trackIT-Systems/pyradiotracking/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-GPL--3.0-green)](LICENSE)

Detect the signals of VHF wildlife tracking transmitters with RTL-SDR receivers. radiotracking is the signal detection of the trackIT vhf:tracker stations (tsOS-vhf), but it runs on any Linux machine with one or more RTL-SDRs.

## How it works

Every SDR runs in its own process:

1. The SDR delivers IQ samples in USB callbacks, which only copy them into a ring buffer.
2. An analysis thread computes a spectrogram of each analysis block (0.3 s by default, segments of `fft_nperseg` samples).
3. In each frequency bin, a signal is a run of segments above `signal_threshold_dbw` and `snr_threshold_db` over the noise level, lasting `signal_min_duration_ms` to `signal_max_duration_ms`. The noise level is the median power of the bin over the block, so it is not raised by the signal itself. Signals may span two blocks.
4. Weaker signals overlapping in time with a stronger one (spectral leakage into neighbouring bins) are dropped as shadows.

The main process collects the signals of all SDRs, groups detections of the same transmission on several SDRs into matched signals, watches and restarts stalled SDRs, and publishes everything.

## Installation

radiotracking needs Python 3.12, 3.13 or 3.14 and librtlsdr. The [trackIT fork of librtlsdr](https://github.com/trackIT-Systems/librtlsdr) is recommended: it provides `rtlsdr_set_tuner_gain_ext` for setting the LNA, mixer and VGA gain stages of the R820T tuner separately. With another librtlsdr, the single `gain` value is used.

```bash
pip install .
```

## Running

```bash
radiotracking --config etc/radiotracking.ini
```

[etc/radiotracking.ini](etc/radiotracking.ini) lists all options with their defaults, `radiotracking -h` describes them. Command line arguments override the config file. Values in the config file are Python literals, e.g. `station = 'name'` and `device = ['0', '1']`.

Signals are published to:

- **CSV files** (`--csv`) in `<path>/<hostname>/radiotracking/`: `<station>_<start time>.csv` for signals, plus `-matched.csv`, `-state.csv` and `-spectrum.csv`.
- **MQTT** (`--mqtt`), as JSON, CSV and CBOR: `<station>/radiotracking/device/<device>/…` for the signals of each SDR, `…/matched/…` for matched signals and `…/state/…` for SDR states. `…/spectrum/cbor` carries the noise level and SNR of every frequency bin of each SDR every `spectrum_interval_s` (300 s), as compact CBOR only. A second instance on the same broker (e.g. for debugging) needs its own `--mqtt-client-id`, otherwise it takes over the session of the first.
- **stdout** (`--sig-stdout`, `--match-stdout`), as CSV.

## Development

```bash
pip install -e . pytest pytest-cov
pytest --cov=radiotracking
```

The tests need neither an SDR nor librtlsdr: if librtlsdr is missing, a stand-in for `rtlsdr` is used. Tests marked `xfail` document known defects. They are strict, so fixing a defect makes its test fail until the marker is removed.

`etc/benchmark.py` times every stage of the analysis per block on synthetic samples (`-s 1200000 -n 1024` for a wide band), e.g. to check the load before deploying on a Pi 4.

GitHub Actions run the tests on every push to `main`: with Python 3.12, 3.13 and 3.14 on the latest dependencies, and with Python 3.13 on the minimum versions (those of Debian 13, which the stations run). The coverage report is shown in the summary of each run.

## Versioning

radiotracking follows [Semantic Versioning](https://semver.org/). Releases are tagged `MAJOR.MINOR.PATCH`, and notable changes are listed in the [changelog](CHANGELOG.md). While the major version is 0, minor releases may contain breaking changes.

## Troubleshooting

**"Failed to allocate zero-copy buffer"**: the kernel limits the memory for USB buffers. Remove the limit with

```bash
echo 0 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
```

**Long signals are not detected**: a signal must end in the block it is analysed in and start at most one block earlier, and it raises the noise level of its block. radiotracking warns at startup if the analysis block is shorter than 6 times `signal_max_duration_ms`; increase `analysis_block_samples` in that case.

**An SDR keeps restarting**: it is restarted if no samples arrive for `sdr_timeout_s`, or if the analysis cannot keep up and the ring buffer overflows. After `sdr_max_restart` restarts, radiotracking terminates, including the SDRs that still work. This is intended: a station with directional antennas needs the signals of all directions, data with a missing direction can't be used for bearings, so the station restarts as a whole (by systemd on tsOS-vhf) instead of continuing incomplete. Check the USB connection and power supply, and the log for the reason.

**The sample rate is adjusted**: RTL-SDRs support [225–300 kHz and 900 kHz–3.2 MHz](https://github.com/osmocom/rtl-sdr/blob/0847e93e0869feab50fd27c7afeb85d78ca04631/src/librtlsdr.c#L1103-L1108). The rate actually set by the device is used for the analysis.
