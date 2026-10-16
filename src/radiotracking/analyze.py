import datetime
import logging
import math
import multiprocessing
import signal
import sys
import threading
import time
from multiprocessing.sharedctypes import Synchronized
from typing import List, Optional, Union

import numpy as np
import rtlsdr
import scipy.fft
import scipy.signal

from radiotracking import Signal, SpectrumMessage, StateMessage, dB, from_dB

logger = logging.getLogger(__name__)


def iq_from_bytes(raw: np.ndarray) -> np.ndarray:
    """
    Convert the bytes of an RTL-SDR (interleaved I and Q) to complex IQ samples.

    The same values as pyrtlsdr's ``read_samples`` (computed in float64) stored as complex64: ``(b - 127.5) / 127.5`` in
    float32 is exact for all 256 byte values, and takes a third of the time of a lookup table.

    Parameters
    ----------
    raw: np.ndarray
        uint8 bytes, flat or one row of I and Q per sample.
    """
    half = np.float32(127.5)
    return ((raw.astype(np.float32) - half) / half).view(np.complex64).reshape(-1)


LN2 = float(np.log(2))
"""Median of an exponential distribution in units of its mean, see :func:`noise_levels`."""


def noise_levels(spectrogram: np.ndarray) -> np.ndarray:
    """
    Noise power of every frequency of a spectrogram.

    The noise power is estimated by the median over time, which is not raised by a signal (a mean would be, by a
    stronger signal and the longer it lasts). The power of a noise-only bin is exponentially distributed, i.e. its
    median is ln(2) times its mean, which is divided out.

    Parameters
    ----------
    spectrogram: np.ndarray
        2d spectrogram data, frequencies by time.
    """
    # the same values as np.median(spectrogram, axis=1): the middle element, or the mean of the two middle elements;
    # partitioned in the memory order of scipy's spectrogram (time by frequencies), which is faster
    by_time = np.ascontiguousarray(spectrogram.T)
    k = by_time.shape[0] // 2
    if by_time.shape[0] % 2:
        noise = np.partition(by_time, k, axis=0)[k] / LN2
    else:
        middle = np.partition(by_time, [k - 1, k], axis=0)
        noise = np.mean(middle[k - 1 : k + 1], axis=0) / LN2

    # a frequency without any noise (the SDR delivering constant samples) would give an infinite SNR and a noise level
    # of -inf dBW, which JSON can't represent; the smallest positive value only replaces an exact zero
    return np.maximum(noise, np.finfo(noise.dtype).tiny)


def spectrogram_of(
    chunk: np.ndarray, sample_rate: float, window: np.ndarray, scale: complex
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Power spectral density of consecutive segments of IQ samples, as ``scipy.signal.spectrogram`` computes it.

    The same operations as scipy (``noverlap=0``, ``return_onesided=False``, ``detrend='constant'``, density scaling), so
    the values are identical, without scipy's input validation and window computation per call.

    Parameters
    ----------
    chunk: np.ndarray
        complex64 IQ samples, a multiple of the segment length
    sample_rate: float
        sample rate (Hz)
    window: np.ndarray
        complex64 window of the segment length, see :func:`spectrogram_window`
    scale: complex
        density scaling of the window, see :func:`spectrogram_window`

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Frequencies (Hz, scipy order 0, positive, negative), segment centers (s) and the spectrogram (frequencies by
        time, a view in scipy's memory order).
    """
    nperseg = len(window)
    segments = chunk.reshape(-1, nperseg)
    segments = segments - np.mean(segments, axis=-1, keepdims=True)
    result = scipy.fft.fft(window * segments, n=nperseg)
    result = np.conjugate(result) * result
    result *= scale
    freqs = scipy.fft.fftfreq(nperseg, 1 / sample_rate)
    times = np.arange(nperseg / 2, len(chunk) - nperseg / 2 + 1, nperseg) / float(sample_rate)
    return freqs, times, result.real.T


def spectrogram_window(window, nperseg: int, sample_rate: float) -> tuple[np.ndarray, complex]:
    """The window (as scipy casts it for complex64 input) and the density scaling for :func:`spectrogram_of`."""
    win = scipy.signal.get_window(window, nperseg).astype(np.complex64)
    return win, 1.0 / (sample_rate * (win * win).sum())

DEFAULT_ANALYSIS_BLOCK_S = 0.3
"""Duration of IQ per analysis block if ``analysis_block_samples`` is not configured."""

SAMPLE_CLOCK_ANCHOR_S = 10.0
"""Seconds of samples after which the sample clock is anchored to the wall clock again, see
:meth:`SignalAnalyzer._anchor_sample_clock`."""

RECOMMENDED_BLOCK_TO_MAX_DURATION = 6
"""The analysis block should be at least this many times as long as the longest accepted signal."""


def resolve_block_samples(sample_rate: float, fft_nperseg: int, configured: int | None = None) -> int:
    """
    Number of IQ samples per analysis block, aligned to (rounded up to a multiple of) ``fft_nperseg``.

    Parameters
    ----------
    sample_rate: float
        The sample rate in Hz.
    fft_nperseg: int
        The number of samples per FFT segment.
    configured: int | None
        The configured ``analysis_block_samples``, None resolves to ``DEFAULT_ANALYSIS_BLOCK_S`` of IQ.
    """
    if fft_nperseg <= 0:
        raise ValueError("fft_nperseg must be positive")
    raw = int(configured) if configured is not None else int(round(DEFAULT_ANALYSIS_BLOCK_S * sample_rate))
    raw = max(fft_nperseg, raw)
    q, r = divmod(raw, fft_nperseg)
    return raw if r == 0 else (q + 1) * fft_nperseg


LIBRTLSDR_DEFAULT_CALLBACK_SAMPLES = 131072
"""Samples per async callback librtlsdr uses if the requested length is invalid (16 * 32 * 512 bytes)."""

CALLBACK_SAMPLES_MULTIPLE = 256
"""librtlsdr only accepts async buffers of a multiple of 512 bytes, i.e. 256 IQ samples, otherwise it uses its default."""


def resolve_callback_length(configured: int | None = None) -> int:
    """
    Number of IQ samples per SDR async callback, as librtlsdr will actually deliver them.

    librtlsdr silently replaces a buffer length that is not a multiple of 512 bytes by its default, so the configured
    length is rounded to the nearest multiple of 256 samples (with a warning), None resolves to the librtlsdr default.

    Parameters
    ----------
    configured: int | None
        The configured ``sdr_callback_length``.
    """
    if configured is None:
        return LIBRTLSDR_DEFAULT_CALLBACK_SAMPLES
    configured = int(configured)
    if configured <= 0:
        raise ValueError("sdr_callback_length must be positive")
    rounded = max(1, round(configured / CALLBACK_SAMPLES_MULTIPLE)) * CALLBACK_SAMPLES_MULTIPLE
    if rounded != configured:
        logger.warning(
            "sdr_callback_length %d is not a multiple of %d samples (librtlsdr would ignore it), using %d.",
            configured,
            CALLBACK_SAMPLES_MULTIPLE,
            rounded,
        )
    return rounded


def resolve_ring_capacity(sample_rate: float, sdr_timeout_s: float, callback_length: int, block_samples: int) -> int:
    """
    Number of IQ samples the ring between the SDR callback and the analysis can hold.

    The ring holds ``sdr_timeout_s`` of IQ, but at least one callback on top of an almost complete analysis block, as
    the analysis only pops complete blocks and the callback pushes complete buffers.
    """
    return max(int(sample_rate * sdr_timeout_s), callback_length + block_samples)


def resolve_sdr_timeout(sdr_timeout_s: int, callback_length: int, sample_rate: float, device: str = "") -> int:
    """
    Stall timeout (s) of an SDR, raised to at least 1.5 times the duration of a callback (in whole seconds).

    The timeout (SIGALRM in the analyzer and the watchdog of the parent) is restarted with every callback, a timeout
    shorter than the time between two callbacks would terminate a working SDR.
    """
    minimum = math.ceil(1.5 * callback_length / sample_rate)
    if sdr_timeout_s >= minimum:
        return sdr_timeout_s
    logger.warning(
        "SDR %s: sdr_timeout_s (%s s) is too short for a callback of %s samples (%.3f s), using %s s.",
        device,
        sdr_timeout_s,
        callback_length,
        callback_length / sample_rate,
        minimum,
    )
    return minimum


def warn_if_block_too_short(block_samples: int, sample_rate: float, signal_max_duration_ms: float) -> bool:
    """
    Log a warning if the analysis block is short compared to the longest accepted signal.

    A signal is only detected if it ends in the block it is analysed in and starts at most one block earlier. The noise
    level of every frequency is estimated per block as well, and the longer a signal is compared to the block, the more
    it raises that estimate, which makes the signal need a higher SNR to be detected.

    Returns
    -------
    bool
        True if a warning was logged.
    """
    block_s = block_samples / sample_rate
    recommended_s = RECOMMENDED_BLOCK_TO_MAX_DURATION * signal_max_duration_ms / 1000
    if block_s >= recommended_s:
        return False

    logger.warning(
        "The analysis block (%d samples, %.3f s) is shorter than %d times signal_max_duration_ms (%g ms, i.e. %.3f s). "
        "Long signals will need a higher SNR to be detected, or can not be detected at all. "
        "Increase analysis_block_samples or lower signal_max_duration_ms.",
        block_samples,
        block_s,
        RECOMMENDED_BLOCK_TO_MAX_DURATION,
        signal_max_duration_ms,
        recommended_s,
    )
    return True


class SignalAnalyzer(multiprocessing.Process):
    """
    Class representing a process for analysis of rtlsdr samples.

    Parameters
    ----------
    device: str
        The device index or serial number of the SDR to use.
    calibration_db: float
        The calibration offset in dB.
    sample_rate: int
        The sample rate of the SDR.
    center_freq: int
        The center frequency of the SDR.
    gain: float
        The gain of the SDR.
    tuner_bandwidth: int
        The R820T2 IF filter bandwidth in Hz. 0 = default (2 * sample_rate is used).
        Must be set together with an appropriate sample_rate to actually capture a wider band.
    fft_nperseg: int
        The number of samples per segment for the FFT.
    fft_window:
        The window function to use for the FFT.
    signal_min_duration_ms: float
        The minimum duration of a signal in ms.
    signal_max_duration_ms: float
        The maximum duration of a signal in ms.
    signal_threshold_dbw: float
        The signal threshold in dBW.
    snr_threshold_db: float
        The SNR threshold in dB.
    verbose: int
        The verbosity level.
    sdr_max_restart: int
        The maximum number of times to restart the SDR.
    sdr_timeout_s: int
        Timeout for stall detection (SIGALRM / parent watchdog) and max ring buffer duration in samples.
    sdr_callback_length: int
        The length of the SDR async read callback in samples (USB stack).
    analysis_block_samples: int | None
        Samples per spectrogram/detection chunk, rounded up to a multiple of fft_nperseg. None uses 0.3 s of IQ.
        Should be at least 6 times signal_max_duration_ms.
    signal_queue: multiprocessing.Queue
        The multiprocessing queue to put the signals in.
    last_data_ts: multiprocessing.Value
        ``time.monotonic()`` of the last USB callback, 0 before the first; read by the watchdog of the parent.
    streaming_ts: multiprocessing.Value
        ``time.monotonic()`` when async USB streaming was entered; 0 until read_bytes_async is about to start.
    spectrum_interval_s: float
        Interval (s of IQ samples) between spectrum messages with the noise and SNR of every frequency; <= 0 disables.
    """

    def __init__(
        self,
        device: str,
        calibration_db: float,
        sample_rate: int,
        center_freq: int,
        gain: float,
        lna_gain: int,
        mixer_gain: int,
        vga_gain: int,
        tuner_bandwidth: int,
        fft_nperseg: int,
        fft_window,
        signal_min_duration_ms: float,
        signal_max_duration_ms: float,
        signal_threshold_dbw: float,
        snr_threshold_db: float,
        verbose: int,
        sdr_max_restart: int,
        sdr_timeout_s: int,
        state_update_s: int,
        sdr_callback_length: int,
        analysis_block_samples: int | None,
        signal_queue: multiprocessing.Queue,
        last_data_ts: Synchronized,
        streaming_ts: Synchronized,
        spectrum_interval_s: float = 300.0,
        **kwargs,
    ):
        super().__init__()

        self.device = device
        self.calibration_db = calibration_db
        # try to use --device as index
        try:
            self.device_index = int(device)
            logger.info(f"Using '{device}' as device index.")
        except ValueError:
            # try to use --device as serial numbers
            try:
                self.device_index = rtlsdr.rtlsdr.RtlSdr.get_device_index_by_serial(device)
                logger.info(f"Using '{device}' as serial number (index: {self.device_index}).")
            except rtlsdr.rtlsdr.LibUSBError:
                logger.warning(f"Device '{device}' could was not found, aborting.")
                sys.exit(1)

        self.sample_rate = sample_rate
        self.center_freq = center_freq
        try:
            self.gain = float(gain)
        except ValueError:
            self.gain = gain

        self.lna_gain = int(lna_gain)
        self.mixer_gain = int(mixer_gain)
        self.vga_gain = int(vga_gain)

        sdr_callback_length = resolve_callback_length(sdr_callback_length)

        self.tuner_bandwidth = tuner_bandwidth
        self.fft_nperseg = fft_nperseg
        self.fft_window = fft_window
        self.signal_min_duration = signal_min_duration_ms / 1000
        self.signal_max_duration = signal_max_duration_ms / 1000
        self.signal_threshold = from_dB(signal_threshold_dbw + calibration_db)
        self.snr_threshold = from_dB(snr_threshold_db)
        self.sdr_callback_length = sdr_callback_length
        self._analysis_block_samples_configured = analysis_block_samples

        self.verbose = verbose

        self.sdr_max_restart = sdr_max_restart
        self.sdr_timeout_s = resolve_sdr_timeout(sdr_timeout_s, sdr_callback_length, sample_rate, device)
        self.state_update_s = state_update_s
        self.spectrum_interval_s = spectrum_interval_s

        self.signal_queue = signal_queue
        self.last_data_ts = last_data_ts
        self.streaming_ts = streaming_ts

        # state shared with the signal handler, which may fire at any point of run(), e.g. while the SDR is still opening
        self.sdr: Optional[rtlsdr.rtlsdr.RtlSdr] = None
        self.last_state: Optional[StateMessage] = None
        self._stop_requested = False

        self._spectrogram_last: None | np.ndarray = None
        self._window: Optional[tuple[np.ndarray, complex]] = None
        self._window_rate: float = 0.0
        self._last_recv_wall: datetime.datetime | None = None

        self.analysis_block_samples: int = 0
        self._ring_capacity: int = 0
        self._ring_buffer: Optional[np.ndarray] = None
        self._ring_read = 0
        self._ring_write = 0
        self._ring_count = 0
        self._ring_cv: Optional[threading.Condition] = None
        self._consumer_stop: Optional[threading.Event] = None
        self._consumer_thread: Optional[threading.Thread] = None
        self._sample_clock_t0: datetime.datetime | None = None
        self._samples_consumed: int = 0
        self._samples_received: int = 0
        self._anchor_min: datetime.datetime | None = None
        self._anchor_samples: int = 0

        self._spectrum_noise: list[np.ndarray] = []
        self._spectrum_peak: Optional[np.ndarray] = None
        self._spectrum_ts: datetime.datetime | None = None
        self._spectrum_samples: int = 0

    def _finalize_analysis_block_samples(self) -> int:
        return resolve_block_samples(self.sample_rate, self.fft_nperseg, self._analysis_block_samples_configured)

    def _r82xx_get_lna_gain(self) -> float:
        r82xx_lna_gain_steps = [0, 9, 13, 40, 38, 13, 31, 22, 26, 31, 26, 14, 19, 5, 35, 13]
        if self.lna_gain < 0 or self.lna_gain >= len(r82xx_lna_gain_steps):
            raise ValueError(
                "lna_gain index %s not in 0 .. 15: 0 == min; see tuner_r82xx.c table r82xx_lna_gain_steps[]" % self.lna_gain
            )

        total_gain = 0
        for i in range(self.lna_gain + 1):
            total_gain += r82xx_lna_gain_steps[i]

        return total_gain / 10

    def _r82xx_get_mixer_gain(self) -> float:
        r82xx_mixer_gain_steps = [0, 5, 10, 10, 19, 9, 10, 25, 17, 10, 8, 16, 13, 6, 3, -8]
        if self.mixer_gain < 0 or self.mixer_gain >= len(r82xx_mixer_gain_steps):
            raise ValueError(
                "mixer_gain index %s not in 0 .. 15: 0 == min; see tuner_r82xx.c table r82xx_mixer_gain_steps[]" % self.mixer_gain
            )

        total_gain = 0
        for i in range(self.mixer_gain + 1):
            total_gain += r82xx_mixer_gain_steps[i]

        return total_gain / 10

    def _r82xx_get_vga_gain(self) -> float:
        r82xx_vga_gain_steps = [0, 26, 26, 30, 42, 35, 24, 13, 14, 32, 36, 34, 35, 37, 35, 36]
        if self.vga_gain < 0 or self.vga_gain >= len(r82xx_vga_gain_steps):
            raise ValueError(
                "vga_gain index %s not in 0 .. 15: 0 == -12 dB; 15 == 40.5 dB; => 3.5 dB/step" % self.vga_gain
            )

        total_gain = 0
        for i in range(self.vga_gain + 1):
            total_gain += r82xx_vga_gain_steps[i]

        return total_gain / 10

    def _r82xx_get_gain(self) -> float:
        VGA_BASE_GAIN = -47

        total_gain = VGA_BASE_GAIN / 10
        total_gain += self._r82xx_get_lna_gain()
        total_gain += self._r82xx_get_mixer_gain()
        total_gain += self._r82xx_get_vga_gain()

        return total_gain

    def _ring_push(self, data: np.ndarray) -> bool:
        assert self._ring_buffer is not None
        n = len(data)
        if self._ring_count + n > self._ring_capacity:
            return False
        w = self._ring_write
        cap = self._ring_capacity
        first = min(n, cap - w)
        self._ring_buffer[w : w + first] = data[:first].astype(self._ring_buffer.dtype, copy=False)
        if first < n:
            self._ring_buffer[: n - first] = data[first:n].astype(self._ring_buffer.dtype, copy=False)
        self._ring_write = (w + n) % cap
        self._ring_count += n
        return True

    def _ring_pop(self, n: int) -> np.ndarray:
        assert self._ring_buffer is not None
        assert self._ring_count >= n
        r = self._ring_read
        cap = self._ring_capacity
        out = np.empty((n, *self._ring_buffer.shape[1:]), dtype=self._ring_buffer.dtype)
        first = min(n, cap - r)
        out[:first] = self._ring_buffer[r : r + first]
        if first < n:
            out[first:] = self._ring_buffer[: n - first]
        self._ring_read = (r + n) % cap
        self._ring_count -= n
        return out

    def _producer_enqueue(self, buffer, context):
        """
        RTL-SDR async callback: copy the bytes into the ring (minimal work, it runs in the USB thread).

        Parameters
        ----------
        buffer:
            The bytes of the callback (interleaved I and Q), a ctypes array or a uint8 array.
        context:
            The context of the callback, unused.
        """
        buffer = np.ctypeslib.as_array(buffer).reshape(-1, 2)
        ts_recv = datetime.datetime.now()
        self._last_recv_wall = ts_recv

        signal.alarm(self.sdr_timeout_s)

        if not self.last_data_ts.value:
            self.update_state(datetime.datetime.now(), StateMessage.State.STARTED)
        else:
            self.update_state(ts_recv, StateMessage.State.RUNNING)
        # monotonic: the watchdog must not be affected by steps of the wall clock (e.g. by NTP)
        self.last_data_ts.value = time.monotonic()
        logger.info("SDR %s received data at %s", self.device, ts_recv)

        assert self._ring_cv is not None
        assert self._consumer_stop is not None
        ring_overflow = False
        with self._ring_cv:
            self._anchor_sample_clock(ts_recv, len(buffer))
            if not self._ring_push(buffer):
                logger.critical(
                    "SDR %s sample ring overflow (capacity %s samples ~%.3f s, occupied %s, "
                    "incoming %s, analysis_block_samples=%s, sdr_callback_length=%s). Stopping.",
                    self.device,
                    self._ring_capacity,
                    self._ring_capacity / self.sample_rate,
                    self._ring_count,
                    len(buffer),
                    self.analysis_block_samples,
                    self.sdr_callback_length,
                )
                self._consumer_stop.set()
                ring_overflow = True
                self._ring_cv.notify_all()
            else:
                self._ring_cv.notify_all()

        if ring_overflow and self.sdr is not None:
            self.update_state(datetime.datetime.now(), StateMessage.State.STOPPED)
            self.sdr.cancel_read_async()

    def _anchor_sample_clock(self, ts_recv: datetime.datetime, n_samples: int):
        """
        Anchor the sample clock (the wall time of the first sample of the stream) to the wall clock.

        Signal timestamps are counted in samples from the first sample, which drifts against the wall clock with the
        error of the SDR's crystal (~35 ms per day at 0.4 ppm, seconds at tens of ppm) and loses lost samples. Every
        callback gives a candidate for the first sample's time: the time it was received minus all samples received.
        The latency of the USB transfer only makes a candidate later, so the earliest of a window of
        ``SAMPLE_CLOCK_ANCHOR_S`` becomes the new anchor; the first callback anchors the clock right away.

        Parameters
        ----------
        ts_recv: datetime.datetime
            wall time the callback was received
        n_samples: int
            samples of the callback
        """
        self._samples_received += n_samples
        candidate = ts_recv - datetime.timedelta(seconds=self._samples_received / self.sample_rate)
        if self._sample_clock_t0 is None:
            self._sample_clock_t0 = candidate
        if self._anchor_min is None or candidate < self._anchor_min:
            self._anchor_min = candidate
        self._anchor_samples += n_samples
        if self._anchor_samples >= SAMPLE_CLOCK_ANCHOR_S * self.sample_rate:
            self._sample_clock_t0 = self._anchor_min
            self._anchor_min = None
            self._anchor_samples = 0

    def _analysis_consumer_loop(self):
        assert self._ring_cv is not None
        assert self._consumer_stop is not None
        while True:
            with self._ring_cv:
                while not self._consumer_stop.is_set() and self._ring_count < self.analysis_block_samples:
                    self._ring_cv.wait(timeout=0.5)
                if self._consumer_stop.is_set() and self._ring_count < self.analysis_block_samples:
                    break
                raw = self._ring_pop(self.analysis_block_samples)
                # the anchor is updated by the callback, read it together with the samples
                sample_clock_t0 = self._sample_clock_t0
            chunk = iq_from_bytes(raw)

            if sample_clock_t0 is None:
                sample_clock_t0 = self._sample_clock_t0 = datetime.datetime.now()

            ts_start = sample_clock_t0 + datetime.timedelta(seconds=self._samples_consumed / self.sample_rate)
            self._samples_consumed += len(chunk)

            # the first milliseconds of a stream are a broadband transient (~18 dB over the noise for 5-7 ms on R820T
            # sticks), too short for a signal, but it would raise the peak of every frequency of the first interval
            n_kept, n_raw, timings = self._analyze_block(chunk, ts_start, spectrum=self._samples_consumed > len(chunk))

            buffer_len_dt = datetime.timedelta(seconds=len(chunk) / self.sample_rate)
            with self._ring_cv:
                backlog_s = self._ring_count / self.sample_rate
            logger.info(
                f"SDR {self.device} analyzed {len(chunk)} samples, "
                f"filtered {n_kept} / {n_raw} signals, "
                f"chunk len: {buffer_len_dt.total_seconds() * 1000:.1f} ms, "
                f"ring backlog: {backlog_s:.3f} s, "
                f"compute: {sum(timings.values()) * 1000:.1f} ms"
            )

            logger.debug(
                f"timings - spectogram: {timings['spectrogram'] * 1000:.1f} ms, "
                + f"noise: {timings['noise'] * 1000:.1f} ms, "
                + f"extract: {timings['extract'] * 1000:.1f} ms, "
                + f"spectrum: {timings['spectrum'] * 1000:.1f} ms, "
                + f"consume: {timings['consume'] * 1000:.1f} ms"
            )

    def _analyze_block(
        self, chunk: np.ndarray, ts_start: datetime.datetime, spectrum: bool = True
    ) -> tuple[int, int, dict[str, float]]:
        """
        Detect and publish the signals of an analysis block, and add it to the spectrum.

        Parameters
        ----------
        chunk: np.ndarray
            IQ samples of the block
        ts_start: datetime.datetime
            start of the block
        spectrum: bool
            whether the block is added to the spectrum message

        Returns
        -------
        tuple[int, int, dict[str, float]]
            The number of published signals, of signals before the shadow filter, and the time (s) of every stage.
        """
        bench_start = time.perf_counter()
        # the window depends on the sample rate, which the hardware may adjust after the analyzer was created
        if self._window is None or self._window_rate != self.sample_rate:
            self._window = spectrogram_window(self.fft_window, self.fft_nperseg, self.sample_rate)
            self._window_rate = self.sample_rate
        freqs, times, spectrogram = spectrogram_of(chunk, self.sample_rate, *self._window)
        bench_spectrogram = time.perf_counter()

        noise = noise_levels(spectrogram)
        bench_noise = time.perf_counter()
        signals, n_raw = self.detect_signals(freqs, times, spectrogram, ts_start, noise)
        bench_extract = time.perf_counter()

        if spectrum:
            self._accumulate_spectrum(freqs, spectrogram, noise, ts_start, len(chunk))
        bench_spectrum = time.perf_counter()

        for s in signals:
            self.consume_signal(s)
        bench_consume = time.perf_counter()

        self._spectrogram_last = spectrogram
        return (
            len(signals),
            n_raw,
            {
                "spectrogram": bench_spectrogram - bench_start,
                "noise": bench_noise - bench_spectrogram,
                "extract": bench_extract - bench_noise,
                "spectrum": bench_spectrum - bench_extract,
                "consume": bench_consume - bench_spectrum,
            },
        )

    def run(self):
        """
        Starts the analyzing process and hands control flow over to rtlsdr.
        """
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)

        # logging levels increase in steps of 10, start with warning
        logging_level = max(0, logging.WARN - (self.verbose * 10))
        logging.basicConfig(level=logging_level)

        # setup sdr
        try:
            sdr = rtlsdr.rtlsdr.RtlSdr(self.device_index)
            sdr.sample_rate = self.sample_rate
            # update configured sample rate with technically possible rate computed by library
            if self.sample_rate != sdr.sample_rate:
                logger.info("adjusting sample rate according to hardware properties: %s", sdr.sample_rate)
                self.sample_rate = sdr.sample_rate
            sdr.center_freq = self.center_freq

            effective_bandwidth = self.tuner_bandwidth if self.tuner_bandwidth != 0 else 2 * self.sample_rate
            sdr.bandwidth = effective_bandwidth
            logger.info(
                "SDR %s: tuner bandwidth set to %s Hz (applied: %s Hz)",
                self.device,
                effective_bandwidth,
                sdr.bandwidth,
            )
        except rtlsdr.rtlsdr.LibUSBError as err:
            logger.warning("SDR %s: could not open or configure (%s).", self.device, err)
            sys.exit(1)

        # terminated while opening the device: do not start streaming
        if self._stop_requested:
            logger.warning("SDR %s: terminated during setup.", self.device)
            sdr.close()
            return

        try:
            if self.lna_gain not in range(0, 16):
                raise ValueError(
                    "lna_gain index %s not in 0 .. 15: 0 == min; see tuner_r82xx.c table r82xx_lna_gain_steps[]" % self.lna_gain
                )
            if self.mixer_gain not in range(0, 16):
                raise ValueError(
                    "mixer_gain index %s not in 0 .. 15: 0 == min; see tuner_r82xx.c table r82xx_mixer_gain_steps[]" % self.mixer_gain
                )
            if self.vga_gain not in range(0, 16):
                raise ValueError(
                    "vga_gain index %s not in 0 .. 15: 0 == -12 dB; 15 == 40.5 dB; => 3.5 dB/step" % self.vga_gain
                )
            sdr.set_manual_gain_enabled(True)
            sdr.set_agc_mode(False)

            ret = rtlsdr.librtlsdr.rtlsdr_set_tuner_gain_ext(sdr.dev_p, self.lna_gain, self.mixer_gain, self.vga_gain)
            if ret == 0:
                self.gain = self._r82xx_get_gain()

                logger.warning(
                    "Gain set manually (lna_index=%s, mixer_index=%s, vga_index=%s, gain=%s, lna_gain: %s, mixer_gain: %s, vga_gain: %s, sdr.gain: %s)",
                    self.lna_gain,
                    self.mixer_gain,
                    self.vga_gain,
                    self.gain,
                    self._r82xx_get_lna_gain(),
                    self._r82xx_get_mixer_gain(),
                    self._r82xx_get_vga_gain(),
                    sdr.gain,
                )
            else:
                raise RuntimeError("Failed setting gain using rtlsdr_set_tuner_gain_ext (%s)" % ret)
        except (AttributeError, ValueError, RuntimeError) as err:
            logger.warning("Error setting gain manually: %s", err)
            sdr.gain = float(self.gain)
            logger.warning("rtlsdr_set_tuner_gain_ext failed, set gain to %s", sdr.gain)

        self.sdr = sdr

        self.analysis_block_samples = self._finalize_analysis_block_samples()
        self._ring_capacity = resolve_ring_capacity(
            self.sample_rate, self.sdr_timeout_s, self.sdr_callback_length, self.analysis_block_samples
        )
        logger.info(
            "SDR %s: analysis_block_samples=%s (~%.3f s), ring_capacity=%s (~%.3f s), sdr_callback_length=%s",
            self.device,
            self.analysis_block_samples,
            self.analysis_block_samples / self.sample_rate,
            self._ring_capacity,
            self._ring_capacity / self.sample_rate,
            self.sdr_callback_length,
        )

        # the bytes of the SDR, one row of I and Q per sample, converted to IQ per analysis block
        self._ring_buffer = np.empty((self._ring_capacity, 2), dtype=np.uint8)
        self._ring_read = 0
        self._ring_write = 0
        self._ring_count = 0
        self._ring_cv = threading.Condition()
        self._consumer_stop = threading.Event()
        self._sample_clock_t0 = None
        self._samples_consumed = 0
        self._samples_received = 0
        self._anchor_min = None
        self._anchor_samples = 0
        self._spectrum_noise = []
        self._spectrum_peak = None
        self._spectrum_ts = None
        self._spectrum_samples = 0

        self._consumer_thread = threading.Thread(target=self._analysis_consumer_loop, name=f"SDR{self.device}-analyze")
        self._consumer_thread.start()

        signal.signal(signal.SIGALRM, self.handle_signal)
        signal.alarm(self.sdr_timeout_s)
        self.streaming_ts.value = time.monotonic()

        try:
            self.sdr.read_bytes_async(self._producer_enqueue, 2 * self.sdr_callback_length)
        except rtlsdr.rtlsdr.LibUSBError as err:
            # librtlsdr may report the cancellation of the transfers as error
            if not self._stop_requested:
                raise
            logger.info("SDR %s: streaming ended after stop request (%s).", self.device, err)
        finally:
            if self._consumer_stop is not None:
                self._consumer_stop.set()
            if self._ring_cv is not None:
                with self._ring_cv:
                    self._ring_cv.notify_all()
            if self._consumer_thread is not None:
                self._consumer_thread.join(timeout=10.0)
                if self._consumer_thread.is_alive():
                    logger.warning("SDR %s: analysis consumer thread did not exit within timeout.", self.device)
            if self.sdr is not None:
                self.sdr.close()

    def handle_signal(self, sig, frame):
        """
        Handles the SIGINT and SIGTERM signals.

        Parameters
        ----------
        sig: int
            The signal number.
        frame:
            The stack frame.
        """
        if sig == signal.SIGALRM:
            ago = "(no signal yet)"
            if self._last_recv_wall is not None:
                ago = f"{(datetime.datetime.now() - self._last_recv_wall).total_seconds():.3f} s ago"
            logger.warning(
                "SDR %s received SIGALRM, last USB callback data %s.",
                self.device,
                ago,
            )
        elif sig == signal.SIGTERM:
            logger.warning("SDR %s received SIGTERM, terminating.", self.device)
        elif sig == signal.SIGINT:
            return

        self._stop_requested = True
        # cancelling may take a while, a pending stall timeout must not cancel a second time
        signal.alarm(0)
        self.update_state(datetime.datetime.now(), StateMessage.State.STOPPED)
        if self._consumer_stop is not None:
            self._consumer_stop.set()
            if self._ring_cv is not None:
                with self._ring_cv:
                    self._ring_cv.notify_all()
        if self.sdr is not None:
            self.sdr.cancel_read_async()

    def update_state(
        self,
        ts: datetime.datetime,
        state: StateMessage.State,
    ):
        # skip update if there is a state
        if self.last_state:
            # the state is different
            if self.last_state.state == state:
                # the state's timeout isn't over
                if self.last_state.ts + datetime.timedelta(seconds=self.state_update_s) >= ts.astimezone():
                    return

        self.last_state = StateMessage(
            self.device,
            ts.astimezone(),
            state,
        )
        self.signal_queue.put(self.last_state)

    def _accumulate_spectrum(
        self, freqs: np.ndarray, spectrogram: np.ndarray, noise: np.ndarray, ts_start: datetime.datetime, n_samples: int
    ):
        """
        Collect the noise and peak power of every frequency of a block, publish a spectrum message per interval.

        The interval is counted in IQ samples, i.e. on the same clock as the signal timestamps. The noise of the interval
        is the median of the noise levels of its blocks, the SNR the ratio of the strongest power to that noise.

        Parameters
        ----------
        freqs: np.ndarray
            spectrogram frequency offsets (scipy order: 0, positive, negative)
        spectrogram: np.ndarray
            2d spectrogram data of the block
        noise: np.ndarray
            noise power of every frequency of the block, see :func:`noise_levels`
        ts_start: datetime.datetime
            start of the block
        n_samples: int
            IQ samples of the block
        """
        if self.spectrum_interval_s <= 0:
            return

        peak = np.max(spectrogram, axis=1)
        if self._spectrum_ts is None:
            self._spectrum_ts = ts_start
            self._spectrum_peak = peak
        else:
            self._spectrum_peak = np.maximum(self._spectrum_peak, peak)
        self._spectrum_noise.append(noise.astype(np.float32))
        self._spectrum_samples += n_samples

        if self._spectrum_samples < self.spectrum_interval_s * self.sample_rate:
            return

        interval_noise = np.median(np.array(self._spectrum_noise), axis=0)
        snr = self._spectrum_peak / interval_noise
        order = np.argsort(freqs)
        self.signal_queue.put(
            SpectrumMessage(
                self.device,
                self._spectrum_ts.astimezone(),
                datetime.timedelta(seconds=self._spectrum_samples / self.sample_rate),
                self.center_freq + float(freqs[order[0]]),
                self.sample_rate / self.fft_nperseg,
                dB(interval_noise[order]) - self.calibration_db,
                dB(snr[order]),
            )
        )

        self._spectrum_noise = []
        self._spectrum_peak = None
        self._spectrum_ts = None
        self._spectrum_samples = 0

    def consume_signal(self, signal: Signal):
        """
        Puts a detected signals into the signal queue.

        Parameters
        ----------
        signal: radiotracking.Signal
            The signal to put into the signal queue.
        """
        logger.debug("SDR %s received %s", self.device, signal)
        self.signal_queue.put(signal)

    @staticmethod
    def is_shadow_of(sig: Signal, signals: List[Signal]) -> Union[None, int]:
        """Compute shadow status of received signals.
        A shadow signal occurs at the same datetime, but with lower power, often in neighbour frequencies.

        Parameters
        ----------
        sig: radiotracking.Signal
            The signal to analyse.
        signals: typing.List[radiotracking.Signal]
            List of signals to compare to.

        Returns
        -------
        Union[None, int]:
            index in signals list, if a shadow of another signal; None if not a shadow.
        """
        # iterate through all other signals
        for i, fsig in enumerate(signals):
            # if sig starts after fsig ends, ignore
            if sig.ts > fsig.ts + fsig.duration:
                continue

            # if sig ends before fsig starts, ignore
            if sig.ts + sig.duration < fsig.ts:
                continue

            # if fsig is louder, we are a shadow, return index
            if fsig.max > sig.max:
                return i

        return None

    def filter_shadow_signals(self, signals: List[Signal]):
        """
        Filters out signals that are too close to each other.

        Parameters
        ----------
        signals: typing.List[radiotracking.Signal]
            The signals to filter.
        """

        signals_status = [SignalAnalyzer.is_shadow_of(sig, signals) for sig in signals]
        logger.debug("shadow list: %s", signals_status)

        return [sig for sig, shadow in zip(signals, signals_status) if shadow is None]

    def _signal_runs(self, spectrogram: np.ndarray, noise: np.ndarray):
        """
        Find the runs of segments over the thresholds that end in this block and have a valid duration.

        A run is a signal candidate of one frequency: consecutive segments over ``signal_threshold`` and
        ``snr_threshold`` over the noise of the frequency. A run may start in the previous block (negative indexes), the
        segments of the previous block are compared to the noise of this block. A run that reaches the end of this block
        might continue, it is left to the next block.

        Returns
        -------
        tuple
            Arrays of the frequency index, the first segment, the segment after the run (relative to this block), the
            start time, the duration (s) and the maximum power of every run.
        """
        last = self._spectrogram_last
        n_last = 0 if last is None else last.shape[1]
        n_bins = spectrogram.shape[0]

        # segments by frequencies: the memory order of scipy's spectrogram, a copy in frequency order costs as much as
        # the whole search
        full = (spectrogram.T if last is None else np.concatenate((last.T, spectrogram.T), axis=0)).ravel()

        # the same comparisons as the per segment search did (negated, which also keeps its behaviour for NaN); the
        # threshold first, so that the SNR is only computed for the few segments over it
        idx = np.flatnonzero(~(full < self.signal_threshold))
        fi = idx % n_bins
        power = full[idx]
        passing = ~(power / noise[fi] < self.snr_threshold)
        idx, fi, power = idx[passing], fi[passing], power[passing]

        # runs of consecutive passing segments of a frequency, in the order frequency, then time: sorted by setting them
        # in a mask in that order, which takes constant time, while sorting the indexes is slow for many noise peaks
        n_segments = len(full) // n_bins
        sorted_mask = np.zeros(n_bins * n_segments, dtype=bool)
        sorted_mask[fi * n_segments + idx // n_bins] = True
        key = np.flatnonzero(sorted_mask)
        fi, segment = key // n_segments, key % n_segments
        power = full[segment * n_bins + fi]
        breaks = (np.diff(segment) != 1) | (np.diff(fi) != 0)
        first = np.concatenate(([True], breaks))[: len(segment)]
        final = np.concatenate((breaks, [True]))[: len(segment)]
        peak = np.maximum.reduceat(power, np.flatnonzero(first)) if len(power) else power
        fi = fi[first]
        start = segment[first] - n_last
        end = segment[final] + 1 - n_last

        # runs entirely in the previous block were reported there, runs reaching the end of this block are deferred
        keep = (end >= 0) & (end < spectrogram.shape[1])

        # the segment duration is known from the configuration, a block may hold a single segment
        segment_duration = self.fft_nperseg / self.sample_rate
        start_dt = start * segment_duration
        duration_s = end * segment_duration - start_dt
        keep &= ~(duration_s < self.signal_min_duration) & ~(duration_s > self.signal_max_duration)

        return fi[keep], start[keep], end[keep], start_dt[keep], duration_s[keep], peak[keep]

    def _signal(
        self,
        freqs: np.ndarray,
        spectrogram: np.ndarray,
        noise: np.ndarray,
        ts_start: datetime.datetime,
        fi: int,
        start: int,
        end: int,
        start_dt: float,
        duration_s: float,
    ) -> Signal:
        """Create the signal of a run, see :meth:`_signal_runs`."""
        last = self._spectrogram_last
        freq_avg = noise[fi]
        if start < 0:
            data = np.concatenate((last[fi][start:], spectrogram[fi][:end]))
        else:
            data = spectrogram[fi][start:end]

        avg = np.mean(data)
        return Signal(
            self.device,
            (ts_start + datetime.timedelta(seconds=start_dt)).astimezone(),
            freqs[fi] + self.center_freq,
            datetime.timedelta(seconds=duration_s),
            dB(np.max(data)) - self.calibration_db,
            dB(avg) - self.calibration_db,
            np.std(dB(data)),
            dB(freq_avg) - self.calibration_db,
            dB(avg / freq_avg),
        )

    def extract_signals(
        self,
        freqs: np.ndarray,
        times: np.ndarray,
        spectrogram: np.ndarray,
        ts_start: datetime.datetime,
        noise: Optional[np.ndarray] = None,
    ) -> List[Signal]:
        """Extract plateaus from spectogram data.

        A signal is a run of segments of a frequency over ``signal_threshold`` and ``snr_threshold`` over the noise of
        the frequency, lasting ``signal_min_duration`` to ``signal_max_duration``. It ends in this block, and may start
        in the previous one.

        Parameters
        ----------
        freqs: np.ndarray
            spectogram frequency offsets
        times: np.ndarray
            spectogram discrete times
        spectrogram: np.ndarray
            2d spectrogram data
        ts_start:
            spectogram start time
        noise: np.ndarray | None
            noise power of every frequency, see :func:`noise_levels`; computed if None

        Returns
        -------
        List[radiotracking.Signal]
            List of signals extracted from spectogram, by frequency and time.
        """
        if len(times) == 0:
            return []

        if noise is None:
            noise = noise_levels(spectrogram)

        runs = self._signal_runs(spectrogram, noise)
        return [
            self._signal(freqs, spectrogram, noise, ts_start, int(f), int(a), int(b), float(t), float(d))
            for f, a, b, t, d in zip(*runs[:5])
        ]

    def detect_signals(
        self,
        freqs: np.ndarray,
        times: np.ndarray,
        spectrogram: np.ndarray,
        ts_start: datetime.datetime,
        noise: Optional[np.ndarray] = None,
    ) -> tuple[List[Signal], int]:
        """
        Extract the signals of a spectrogram without their shadows.

        The same as :meth:`filter_shadow_signals` of :meth:`extract_signals`, but the shadows are dropped before signals
        are created: a strong signal can leak into all frequencies, i.e. hundreds of shadows per block.

        Returns
        -------
        tuple[List[radiotracking.Signal], int]
            The signals that are no shadow, and the number of signals before the shadow filter.
        """
        if len(times) == 0:
            return [], 0

        if noise is None:
            noise = noise_levels(spectrogram)

        fi, start, end, start_dt, duration_s, peak = self._signal_runs(spectrogram, noise)
        if len(fi) == 0:
            return [], 0

        # maximum power of every run, as the signals will report it
        max_dbw = dB(peak) - self.calibration_db

        # start and end in microseconds after ts_start, rounded like the timestamps and durations of the signals
        us = datetime.timedelta(microseconds=1)
        ts_us = np.array([datetime.timedelta(seconds=float(t)) // us for t in start_dt], dtype=np.int64)
        end_us = ts_us + np.array([datetime.timedelta(seconds=float(d)) // us for d in duration_s], dtype=np.int64)

        # a signal is a shadow if a louder one overlaps in time (see is_shadow_of), compared in chunks to bound memory
        shadow = np.zeros(len(fi), dtype=bool)
        for i in range(0, len(fi), 1024):
            rows = slice(i, i + 1024)
            overlap = ~(ts_us[rows, None] > end_us[None, :]) & ~(end_us[rows, None] < ts_us[None, :])
            shadow[rows] = np.any(overlap & (max_dbw[None, :] > max_dbw[rows, None]), axis=1)

        kept = [
            self._signal(freqs, spectrogram, noise, ts_start, int(f), int(a), int(b), float(t), float(d))
            for f, a, b, t, d in zip(fi[~shadow], start[~shadow], end[~shadow], start_dt[~shadow], duration_s[~shadow])
        ]
        return kept, len(fi)
