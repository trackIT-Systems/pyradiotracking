# Changelog

All notable changes to [pyradiotracking](https://github.com/trackIT-Systems/pyradiotracking) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the major version is 0, minor releases may contain breaking changes.

Releases up to 0.3.0 were not tagged consistently; the entries for those versions and for the history before them are reconstructed from the git log.

## [Unreleased]

### Added

- Test suite (`pytest`) covering messages, matching, configuration, consumers, signal analysis and the runner; known defects are recorded as strict `xfail` tests
- Sample rate (`--sample-rate`) and tuner bandwidth (`--tuner-bandwidth`) are configurable, tuner bandwidth defaults to twice the sample rate
- SDR spectrogram metrics are published in state messages (`--status-interval`)

### Fixed

- Analyzers that hang after opening the device before the first USB sample are restarted
- The `[optional arguments]` section (`verbose`, `calibrate`, `station`, `schedule`, `config`) was ignored since Python 3.10, which names the group `[options]`. Config files are now read with either section name, if both are present `[options]` wins; written configs use the name of the running Python

### Removed

- Web dashboard (`--dashboard`, `--dashboard-host`, `--dashboard-port`, `--dashboard-signals` and the `[dashboard]` config section) and the configuration web UI (`python3 -m radiotracking.config`)
- `radiotracking.present` module, and the `dash` and `matplotlib` dependencies
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
