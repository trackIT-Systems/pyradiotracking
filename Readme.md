pyradiotracking
===

Detect signals of wildlife tracking systems with RTL SDR devices.

### Installation

radiotracking requires Python >= 3.11 and [librtlsdr](https://github.com/trackIT-Systems/librtlsdr) (via [pyrtlsdr](https://github.com/pyrtlsdr/pyrtlsdr)).

```bash
$ pip install .
```

### Usage

An example configuration with all options is provided in [etc/radiotracking.ini](etc/radiotracking.ini). Command line options override values from the configuration file.

```bash
$ radiotracking --config etc/radiotracking.ini --csv --path data
$ radiotracking -h
usage: radiotracking [-h] [-v] [--calibrate] [--config CONFIG] [--station STATION] [--schedule [SCHEDULE ...]] [-d [DEVICE ...]]
                     [-c [CALIBRATION ...]] [-f CENTER_FREQ] [-s SAMPLE_RATE] [-b SDR_CALLBACK_LENGTH] [-g GAIN] [--lna-gain LNA_GAIN]
                     [--mixer-gain MIXER_GAIN] [--vga-gain VGA_GAIN] [--tuner-bandwidth TUNER_BANDWIDTH] [--sdr-max-restart SDR_MAX_RESTART]
                     [--sdr-timeout-s SDR_TIMEOUT_S] [--sdr-dynamic-scheduling] [--state-update-s STATE_UPDATE_S]
                     [--status-interval STATUS_INTERVAL] [-n FFT_NPERSEG] [-w FFT_WINDOW] [-t SIGNAL_THRESHOLD_DBW] [-r SNR_THRESHOLD_DB]
                     [-l SIGNAL_MIN_DURATION_MS] [-u SIGNAL_MAX_DURATION_MS] [--analysis-block-samples ANALYSIS_BLOCK_SAMPLES]
                     [--matching-timeout-s MATCHING_TIMEOUT_S] [-mt MATCHING_TIME_DIFF_S] [-mb MATCHING_BANDWIDTH_HZ]
                     [-md MATCHING_DURATION_DIFF_MS] [--sig-stdout] [--match-stdout] [--path PATH] [--csv] [--export-config] [--mqtt]
                     [--mqtt-host MQTT_HOST] [--mqtt-port MQTT_PORT] [--mqtt-qos MQTT_QOS] [--mqtt-keepalive MQTT_KEEPALIVE] [-mv]

Detect signals of wildlife tracking systems with RTL SDR devices

options:
  -h, --help            show this help message and exit
  -v, --verbose         increase output verbosity (default: 0)
  --calibrate           enable calibration mode (default: False)
  --config CONFIG       configuration file (default: etc/radiotracking.ini)
  --station STATION     name of the station (default: <hostname>)
  --schedule [SCHEDULE ...]
                        specify a schedule of operation, e.g. 18:00-18:59:59 (default: [])

rtl-sdr:
  -d, --device [DEVICE ...]
                        device indexes or names (default: ['0'])
  -c, --calibration [CALIBRATION ...]
                        device calibration gain (dB) (default: [])
  -f, --center-freq CENTER_FREQ
                        center frequency to tune to (Hz) (default: 150150000)
  -s, --sample-rate SAMPLE_RATE
                        sample rate (Hz) (default: 300000)
  -b, --sdr-callback-length SDR_CALLBACK_LENGTH
                        number of samples to read per batch (default: None)
  -g, --gain GAIN       gain, supported levels 0.0 - 49.6 (default: 49.6)
  --lna-gain LNA_GAIN   lna_gain index in 0 .. 15 (default: 15)
  --mixer-gain MIXER_GAIN
                        mixer_gain index in 0 .. 15 (default: 14)
  --vga-gain VGA_GAIN   vga_gain index in 0 .. 15 (default: 15)
  --tuner-bandwidth TUNER_BANDWIDTH
                        R820T2 IF filter bandwidth (Hz); 0 = default (2 * sample-rate) (default: 0)
  --sdr-max-restart SDR_MAX_RESTART
                        maximal restart count per SDR device (default: 3)
  --sdr-timeout-s SDR_TIMEOUT_S
                        Stall timeout (s) for SIGALRM/parent watchdog; also caps IQ ring size at sample_rate*timeout samples (default: 2)
  --sdr-dynamic-scheduling
                        allow the OS to schedule SDR processes freely (default: pin each SDR to a CPU, device index modulo ncpu) (default: False)
  --state-update-s STATE_UPDATE_S
                        interval in which state update messages should be recorded (s) (default: 300)
  --status-interval STATUS_INTERVAL
                        interval for publishing aggregated spectrogram metrics in state messages (s); <=0 disables (default: 60.0)

analysis:
  -n, --fft-nperseg FFT_NPERSEG
                        fft number of samples (default: 256)
  -w, --fft-window FFT_WINDOW
                        fft window function (default: 'hamming')
  -t, --signal-threshold-dbw SIGNAL_THRESHOLD_DBW
                        lower limit for signal intensity (dBW) (default: -90.0)
  -r, --snr-threshold-db SNR_THRESHOLD_DB
                        lower limit for signal-to-noise ratio (dB) (default: 5.0)
  -l, --signal-min-duration-ms SIGNAL_MIN_DURATION_MS
                        lower limit for signal duration (ms) (default: 8)
  -u, --signal-max-duration-ms SIGNAL_MAX_DURATION_MS
                        upper limit for signal duration (ms) (default: 40)
  --analysis-block-samples ANALYSIS_BLOCK_SAMPLES
                        IQ samples per spectrogram/detection chunk (default: ~0.1 s at sample_rate, aligned to fft_nperseg) (default: None)

matching:
  --matching-timeout-s MATCHING_TIMEOUT_S
                        timeout for adding signals to a match group (default: 2.0)
  -mt, --matching-time-diff-s MATCHING_TIME_DIFF_S
                        error margin for timestamp matching (s) (default: 0)
  -mb, --matching-bandwidth-hz MATCHING_BANDWIDTH_HZ
                        error margin for frequency (Hz) (default: 0)
  -md, --matching-duration-diff-ms MATCHING_DURATION_DIFF_MS
                        error margin for duration (ms) (default: None)

publish:
  --sig-stdout          enable stdout signal publishing (default: False)
  --match-stdout        enable stdout matched signals publishing (default: False)
  --path PATH           file output path (default: data)
  --csv                 enable csv data publishing (default: False)
  --export-config       export configuration (default: False)
  --mqtt                enable mqtt data publishing (default: False)
  --mqtt-host MQTT_HOST
                        hostname of mqtt broker (default: localhost)
  --mqtt-port MQTT_PORT
                        port of mqtt broker (default: 1883)
  --mqtt-qos MQTT_QOS   mqtt quality of service level (0, 1, 2) (default: 1)
  --mqtt-keepalive MQTT_KEEPALIVE
                        timeout for mqtt connection (s) (default: 3600)
  -mv, --mqtt-verbose   increase mqtt logging verbosity (default: 0)
```

### Versioning

radiotracking follows [Semantic Versioning](https://semver.org/). Releases are tagged as `MAJOR.MINOR.PATCH`, notable changes are documented in the [changelog](CHANGELOG.md). While the major version is 0, minor releases may contain breaking changes.

### Troubleshooting

#### Failed to allocate zero-copy buffer

The size of zero-copy kernel buffers is limited.

```
Failed to allocate zero-copy buffer for transfer 5
Falling back to buffers in userspace
Failed to submit transfer 6
Please increase your allowed usbfs buffer size with the following command:
echo 0 > /sys/module/usbcore/parameters/usbfs_memory_mb
```
For larger buffers, as preferable for high sampling rates in terms of effificiency, the usbfs memory limit can be removed:

`echo 0 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb`

#### Sample buffer, timestamps, and stalls

USB/async callbacks only copy IQ into an internal ring buffer (capacity `sample_rate * sdr_timeout_s` samples). A separate thread runs the spectrogram and detection on fixed **analysis** chunks (`analysis_block_samples`, default about **0.1 s** of IQ at `sample_rate`, rounded up to a multiple of `fft_nperseg`). This decouples `sdr_callback_length` (USB transfer size) from FFT block size.

Signal timestamps follow a **sample-count timeline** (wall time anchors the first consumed chunk; later chunk starts advance by `samples / sample_rate`). Stall detection uses wall time: child `SIGALRM` / `sdr_timeout_s` after USB callbacks begin, and the parent watchdog if no USB callbacks arrive (startup grace while opening the dongle, then `sdr_timeout_s` after async read starts). Child `SIGALRM` cannot interrupt a hang inside libusb; the parent then `SIGTERM`s and `SIGKILL`s the child.

If the ring fills (analysis cannot keep up), the analyzer stops and the normal per-device restart logic applies. Avoid overload with less expensive analysis settings, or increase `sdr_timeout_s` only if you accept a larger ring backlog.


### Caveats

- The sample rate of RTL-SDR is [limited to certain numbers](https://github.com/osmocom/rtl-sdr/blob/0847e93e0869feab50fd27c7afeb85d78ca04631/src/librtlsdr.c#L1103-L1108) in 225 kHz - 3.2 GHz, excluding 300 kHz - 900 kHz.
