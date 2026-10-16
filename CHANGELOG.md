# Changelog

All notable changes to [pyradiotracking](https://github.com/trackIT-Systems/pyradiotracking) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the major version is 0, minor releases may contain breaking changes.

Releases up to 0.3.0 were not tagged consistently; the entries for those versions and for the history before them are reconstructed from the git log.

## [Unreleased]

### Changed

- The default analysis block (`analysis_block_samples` unset) is 0.3 s of IQ (was 0.1 s). A warning is logged at startup if the block, default or configured, is shorter than 6 times `signal_max_duration_ms`
- `signal_max_duration_ms` defaults to 50 (was 40), tags with up to 45 ms are detected
- The noise level of a frequency is the median of its power over the block (divided by ln 2) instead of the mean. The mean was raised by the signal itself, so signals filling more than about 30 % of the block could not reach `snr_threshold_db` (with the old 0.1 s block: signals longer than ~32 ms were never detected at 5 dB). `noise_dBW` and `snr_dB` of signals are the true values now; weak, long signals are detected more readily
- Python 3.12, 3.13 and 3.14 are supported (was >= 3.11); the test suite runs on all three, with cbor2 5 and 6
- Minimum dependency versions are the oldest tested ones, those of Debian 13 (trixie): numpy 2.2.4, scipy 1.15.3, paho-mqtt 2.1.0, cbor2 5.6.5 and pyrtlsdr 0.5.0 (were numpy 2.0, scipy 1.10.1, which does not support Python 3.12, paho-mqtt 2.0, cbor2 5.6.3 and pyrtlsdr 0.3.0). The test suite also passes with the latest versions (numpy 2.5.4, scipy 1.18.1)
- Signal timestamps are the start of the first segment of the signal (were the center of the segment before it). Together with the fixes below, `ts`, `duration`, `avg_dBW` and `std_dB` are exact to the segment for signals covering whole segments
- Signal timestamps are counted from one callback before the first callback was received, i.e. from the first sample; they were late by one callback (0.44 s by default)
- `sdr_callback_length` defaults to librtlsdr's 131072 samples, which was used anyway: librtlsdr ignored the old default (`sample_rate`, not a multiple of 512 bytes). Other values are rounded to a multiple of 256 samples, with a warning
- The sample ring holds at least one callback and one analysis block, `sdr_timeout_s` is raised to 1.5 times the duration of a callback (with a warning); a too short timeout restarted every SDR before its first callback until the station terminated
- Values of the config file are converted to the option's type (e.g. `device = [0]` is `['0']`, `center_freq = 150.1e6` is an int); a value that is not a python literal or of the wrong type (e.g. `sdr_timeout_s = 2.5`) exits with a message naming the option
- Signals are searched vectorized over all frequencies instead of per frequency in Python, and shadows are dropped before signals are created. The signals are identical (checked against the former search on 2.2 million signals of recorded IQ), the search takes 1.7 instead of 14 ms per block at 1.2 MHz with 1024 bins on a Pi 5, and a close tag leaking into all frequencies no longer creates hundreds of signals per block
- The SDR callback only copies the bytes of the SDR into the sample ring (2 bytes per sample instead of 8), they are converted to IQ per analysis block with float32 arithmetic giving exactly pyrtlsdr's values. The USB thread no longer converts samples to complex128, and the conversion takes 1.5 instead of 4.6 ms per block at 1.2 MHz on a Pi 5
- The spectrogram and the noise level per frequency are computed with the operations of `scipy.signal.spectrogram` and `np.median` (bit-identical, checked by tests), but in the memory order of the spectrogram and without scipy's per-call window computation: 15.7 instead of 17.9 ms per block at 1.2 MHz on a Pi 5

### Added

- Test suite (`pytest`) covering messages, matching, configuration, consumers, signal analysis and the runner; known defects are recorded as strict `xfail` tests
- GitHub Actions run the test suite on every push to `main`, with Python 3.12, 3.13 and 3.14 on the latest dependencies and with Python 3.13 on the minimum versions (those of Debian 13); the coverage report is shown in the summary of each run
- Sample rate (`--sample-rate`) and tuner bandwidth (`--tuner-bandwidth`) are configurable, tuner bandwidth defaults to twice the sample rate
- Spectrum messages with the noise level and the SNR of every frequency bin of an SDR, every `--spectrum-interval-s` (default 300 s) of IQ samples (#3). The noise is the median of the per-frequency noise level the detection uses (not raised by tags), the SNR the strongest power in the interval over it; the first analysis block of a stream is left out, it starts with a broadband transient of 5-7 ms. They are published as compact CBOR only (`<station>/radiotracking/spectrum/cbor`, half precision values, ~1.6 kB for 256 bins) and written to `-spectrum.csv`. They replace the peak, RMS and SNR of the whole band that state messages carried on `main` since May 2026 (`--status-interval`): state messages report the lifecycle only again, a configured `status_interval` is ignored with a warning

### Fixed

- Analyzers that hang after opening the device before the first USB sample are restarted
- Signals counted the segment before their start into the signal: `duration` was one segment (0.85 ms) too long, `avg_dBW` too low and `std_dB` far too high (8 to 10 dB instead of about 1 dB for real tags)
- The start of signals beginning in the previous analysis block was one more segment too early
- Signals ending exactly at the end of an analysis block were lost: they were deferred to the next block, which did not look back for them
- `noise_dBW` did not include the calibration of the device, `max_dBW` and `avg_dBW` did
- On termination, the pending stall timeout cancelled the USB transfers a second time, and the analyzers exited with a `LibUSBError` traceback
- A missing config file and unknown options (e.g. typos) in it are reported, they were ignored silently
- No devices or a device configured more than once exit at startup, a duplicate used to fail and terminate the station after its restarts
- Signal extraction raised `IndexError` (killing the analysis thread, and with it the station after the restarts) when the analysis block was a single FFT segment, e.g. `fft_nperseg` of 32768 at 300 kHz; the segment duration is computed from `fft_nperseg / sample_rate` now. Such a block is still too short to detect signals
- Error messages for invalid `lna_gain`, `mixer_gain` and `vga_gain` values showed the raw format string and a tuple
- `MatchedSignal` printed a power of exactly 0.0 dBW as `None`
- `write_config(help=True)` (`etc/generate-example-config.py`) raised on Python 3.14 and lowercased the help texts on 3.13; configs are written without ConfigParser now, and read without `%` interpolation
- Decoding CBOR messages with `uncborify` (`etc/mqtt-stream.py`) failed with cbor2 6, which calls tag hooks with different arguments
- A consumer (CSV, MQTT, matcher) raising an exception no longer keeps the remaining consumers from receiving the message, the error is logged instead
- A SIGTERM or SIGALRM while the SDR is still opening raised `AttributeError` in the analyzer process; it now stops cleanly without starting to stream
- Matched signals are published after the matching timeout while idle and on shutdown, they were only published when a later signal arrived, so the last detections before a quiet period (or shutdown) were lost
- The `[optional arguments]` section (`verbose`, `calibrate`, `station`, `schedule`, `config`) was ignored since Python 3.10, which names the group `[options]`. Config files are now read with either section name, if both are present `[options]` wins; written configs use the name of the running Python

### Removed

- The unused `setuptools` runtime dependency
- Schedule of operation (`--schedule`, `schedule` config option) and the `schedule` dependency; radiotracking always runs permanently. A `schedule` entry in an existing config file is ignored, a warning is logged if it is not empty
- Publishing log messages via MQTT (`<station>/radiotracking/log/csv`) and `--mqtt-verbose`. Nothing subscribed to them, and with the `fork` start method the SDR processes inherited the handler without the MQTT network thread: their log messages were lost and blocked the process for the 2 s publish timeout each (e.g. 2 s longer startup and shutdown). A `mqtt_verbose` entry in an existing config file is ignored, a warning is logged if it is not 0
- Web dashboard (`--dashboard`, `--dashboard-host`, `--dashboard-port`, `--dashboard-signals` and the `[dashboard]` config section) and the configuration web UI (`python3 -m radiotracking.config`)
- `radiotracking.present` module, and the `dash` and `matplotlib` dependencies
- `ArgConfParser.immutable_args()`, which relied on a private argparse API that changed in Python 3.13 and was only used by the removed config web UI
- `radiotracking.bearing` module (unused `Gottwald2019Bearings` bearing calculation from 2021)

  Config files that still contain a `[dashboard]` section keep working, the section is ignored.

## [0.4.0] - 2026-04-15

### Added

- Ring buffer decouples USB callbacks from analysis, the spectrogram and detection run on fixed analysis chunks (`--analysis-block-samples`)
- Hang-up countermeasures: parent watchdog for analyzers that stop delivering USB data, escalating from `SIGTERM` to `SIGKILL`

## [0.3.3] - 2026-04-14

### Added

- Dynamic SDR / CPU affinity scheduling (`--sdr-dynamic-scheduling`); by default each SDR process is pinned to a CPU

### Changed

- Dashboard updates only reload data instead of redrawing figures
- Radiotracking names updated

### Fixed

- SDR devices are terminated properly, which reduces power draw
- The SDR is initialized before it is opened

## [0.3.2] - 2025-10-29

### Changed

- Migrated to paho-mqtt 2
- numpy >= 2 is required

## [0.3.1] - 2025-06-10

### Fixed

- Timezone handling in state messages (#4)

## [0.3.0] - 2025-03-18

### Added

- Fine-grained manual gain control (`--lna-gain`, `--mixer-gain`, `--vga-gain`), a fixed gain disables the AGC
- Improved logging in manual gain mode

### Changed

- Signals are saved and sent with timezone-aware timestamps
- Packaging moved from `setup.py` to `pyproject.toml` (PDM) with a `src` layout

### Fixed

- Subscripting of `Synchronized` is not allowed in Python 3.11

## Before 0.3.0

Versions before 0.3.0 were not released with a changelog. Notable changes from the git history:

- 2023-04: hardware SDR frequency update fixed (#2)
- 2022-11: `StateMessage` implemented and published via MQTT, with a configurable update interval (#28)
- 2022-07: legacy code removed
- 2022-05: fixed an error on termination (#27)
- 2022-02: MQTT logging handler, automatic gain correction explicitly disabled, docstrings added
- 2021-10: analysis separated from the config runner
- 2021-06: SDR data watchdog
- 2021-03: bearing calculation added
- 2020-10: first implementation (`pybatdetect`): signal detection, consumers, multiple devices and CSV export

[Unreleased]: https://github.com/trackIT-Systems/pyradiotracking/compare/0.4.0...HEAD
[0.4.0]: https://github.com/trackIT-Systems/pyradiotracking/compare/0.3.3...0.4.0
[0.3.3]: https://github.com/trackIT-Systems/pyradiotracking/compare/0.3.2...0.3.3
[0.3.2]: https://github.com/trackIT-Systems/pyradiotracking/compare/0.3.1...0.3.2
[0.3.1]: https://github.com/trackIT-Systems/pyradiotracking/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/trackIT-Systems/pyradiotracking/releases/tag/0.3.0
