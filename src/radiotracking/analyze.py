import datetime
import logging
import multiprocessing
import signal
import sys
import threading
import time
from multiprocessing.sharedctypes import Synchronized
from typing import List, Optional, Union

import numpy as np
import rtlsdr
import scipy.signal

from radiotracking import Signal, StateMessage, dB, from_dB

logger = logging.getLogger(__name__)


def default_analysis_block_samples(
    sample_rate: float, fft_nperseg: int, block_duration_s: float = 0.1
) -> int:
    """
    Default segment-aligned analysis block: ``block_duration_s`` of IQ (rounded), at least one FFT segment.
    """
    if fft_nperseg <= 0:
        raise ValueError("fft_nperseg must be positive")
    raw = max(fft_nperseg, int(round(block_duration_s * sample_rate)))
    q, r = divmod(raw, fft_nperseg)
    return raw if r == 0 else (q + 1) * fft_nperseg


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
        Samples per spectrogram/detection chunk; None uses ~0.1 s of IQ, rounded up to a multiple of fft_nperseg.
    signal_queue: multiprocessing.Queue
        The multiprocessing queue to put the signals in.
    last_data_ts: multiprocessing.Value
        The multiprocessing value to put the last data timestamp in.
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

        if sdr_callback_length is None:
            sdr_callback_length = sample_rate

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
        self.sdr_timeout_s = sdr_timeout_s
        self.state_update_s = state_update_s

        self.signal_queue = signal_queue
        self.last_data_ts = last_data_ts

        self._spectrogram_last: None | np.ndarray = None
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

    def _align_block_to_fft(self, n: int) -> int:
        n = max(self.fft_nperseg, n)
        q, r = divmod(n, self.fft_nperseg)
        if r:
            n = (q + 1) * self.fft_nperseg
        return n

    def _finalize_analysis_block_samples(self) -> int:
        if self._analysis_block_samples_configured is not None:
            return self._align_block_to_fft(int(self._analysis_block_samples_configured))
        return default_analysis_block_samples(self.sample_rate, self.fft_nperseg)

    def _r82xx_get_lna_gain(self) -> float:
        r82xx_lna_gain_steps = [0, 9, 13, 40, 38, 13, 31, 22, 26, 31, 26, 14, 19, 5, 35, 13]
        if self.lna_gain < 0 or self.lna_gain >= len(r82xx_lna_gain_steps):
            raise ValueError(
                "lna_gain index %s not in 0 .. 15: 0 == min; see tuner_r82xx.c table r82xx_lna_gain_steps[]",
                self.lna_gain,
            )

        total_gain = 0
        for i in range(self.lna_gain + 1):
            total_gain += r82xx_lna_gain_steps[i]

        return total_gain / 10

    def _r82xx_get_mixer_gain(self) -> float:
        r82xx_mixer_gain_steps = [0, 5, 10, 10, 19, 9, 10, 25, 17, 10, 8, 16, 13, 6, 3, -8]
        if self.mixer_gain < 0 or self.mixer_gain >= len(r82xx_mixer_gain_steps):
            raise ValueError(
                "mixer_gain index %s not in 0 .. 15: 0 == min; see tuner_r82xx.c table r82xx_mixer_gain_steps[]",
                self.mixer_gain,
            )

        total_gain = 0
        for i in range(self.mixer_gain + 1):
            total_gain += r82xx_mixer_gain_steps[i]

        return total_gain / 10

    def _r82xx_get_vga_gain(self) -> float:
        r82xx_vga_gain_steps = [0, 26, 26, 30, 42, 35, 24, 13, 14, 32, 36, 34, 35, 37, 35, 36]
        if self.vga_gain < 0 or self.vga_gain >= len(r82xx_vga_gain_steps):
            raise ValueError(
                "vga_gain index %s not in 0 .. 15: 0 == -12 dB; 15 == 40.5 dB; => 3.5 dB/step",
                self.vga_gain,
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
        out = np.empty(n, dtype=self._ring_buffer.dtype)
        first = min(n, cap - r)
        out[:first] = self._ring_buffer[r : r + first]
        if first < n:
            out[first:] = self._ring_buffer[: n - first]
        self._ring_read = (r + n) % cap
        self._ring_count -= n
        return out

    def _producer_enqueue(self, buffer: np.ndarray, context):
        """RTL-SDR async callback: enqueue samples into the ring (minimal work)."""
        ts_recv = datetime.datetime.now()
        self._last_recv_wall = ts_recv

        signal.alarm(self.sdr_timeout_s)

        if not self.last_data_ts.value:
            self.update_state(datetime.datetime.now(), StateMessage.State.STARTED)
        else:
            self.update_state(ts_recv, StateMessage.State.RUNNING)
        self.last_data_ts.value = datetime.datetime.timestamp(ts_recv)
        logger.info(f"SDR {self.device} received data at {self.last_data_ts.value}")

        assert self._ring_cv is not None
        assert self._consumer_stop is not None
        ring_overflow = False
        with self._ring_cv:
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

    def _analysis_consumer_loop(self):
        assert self._ring_cv is not None
        assert self._consumer_stop is not None
        while True:
            with self._ring_cv:
                while not self._consumer_stop.is_set() and self._ring_count < self.analysis_block_samples:
                    self._ring_cv.wait(timeout=0.5)
                if self._consumer_stop.is_set() and self._ring_count < self.analysis_block_samples:
                    break
                chunk = self._ring_pop(self.analysis_block_samples)

            if self._sample_clock_t0 is None:
                self._sample_clock_t0 = datetime.datetime.now()

            ts_start = self._sample_clock_t0 + datetime.timedelta(seconds=self._samples_consumed / self.sample_rate)
            self._samples_consumed += len(chunk)

            bench_start = time.time()
            freqs, times, spectrogram = scipy.signal.spectrogram(
                chunk,
                fs=self.sample_rate,
                window=self.fft_window,
                nperseg=self.fft_nperseg,
                noverlap=0,
                return_onesided=False,
            )
            bench_spectrogram = time.time()

            signals = self.extract_signals(freqs, times, spectrogram, ts_start)
            bench_extract = time.time()

            filtered = self.filter_shadow_signals(signals)
            bench_filter = time.time()

            for s in filtered:
                self.consume_signal(s)
            bench_consume = time.time()

            buffer_len_dt = datetime.timedelta(seconds=len(chunk) / self.sample_rate)
            with self._ring_cv:
                backlog_s = self._ring_count / self.sample_rate
            logger.info(
                f"SDR {self.device} analyzed {len(chunk)} samples, "
                f"filtered {len(filtered)} / {len(signals)} signals, "
                f"chunk len: {buffer_len_dt.total_seconds() * 1000:.1f} ms, "
                f"ring backlog: {backlog_s:.3f} s, "
                f"compute: {(bench_consume - bench_start) * 1000:.1f} ms"
            )

            logger.debug(
                f"timings - spectogram: {(bench_spectrogram - bench_start) * 1000:.1f} ms, "
                + f"extract: {(bench_extract - bench_spectrogram) * 1000:.1f} ms, "
                + f"filter: {(bench_filter - bench_extract) * 1000:.1f} ms, "
                + f"consume: {(bench_consume - bench_filter) * 1000:.1f} ms"
            )
            self._spectrogram_last = spectrogram

    def run(self):
        """
        Starts the analyzing process and hands control flow over to rtlsdr.
        """
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)

        # initialize state
        self.last_state = None

        # logging levels increase in steps of 10, start with warning
        logging_level = max(0, logging.WARN - (self.verbose * 10))
        logging.basicConfig(level=logging_level)

        # setup sdr
        sdr = rtlsdr.rtlsdr.RtlSdr(self.device_index)
        sdr.sample_rate = self.sample_rate
        # update configured sample rate with technically possible rate compured by library
        if self.sample_rate != sdr.sample_rate:
            logger.info("adjusting sample rate according to hardware properties: %s", sdr.sample_rate)
            self.sample_rate = sdr.sample_rate
        sdr.center_freq = self.center_freq

        try:
            if self.lna_gain not in range(0, 16):
                raise ValueError(
                    "lna_gain index %s not in 0 .. 15: 0 == min; see tuner_r82xx.c table r82xx_lna_gain_steps[]",
                    self.lna_gain,
                )
            if self.mixer_gain not in range(0, 16):
                raise ValueError(
                    "mixer_gain index %s not in 0 .. 15: 0 == min; see tuner_r82xx.c table r82xx_mixer_gain_steps[]",
                    self.mixer_gain,
                )
            if self.vga_gain not in range(0, 16):
                raise ValueError(
                    "vga_gain index %s not in 0 .. 15: 0 == -12 dB; 15 == 40.5 dB; => 3.5 dB/step", self.vga_gain
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
                raise RuntimeError("Failed setting gain using rtlsdr_set_tuner_gain_ext (%s)", ret)
        except (AttributeError, ValueError, RuntimeError) as err:
            logger.warning("Error setting gain manually: %s", err)
            sdr.gain = float(self.gain)
            logger.warning("rtlsdr_set_tuner_gain_ext failed, set gain to %s", sdr.gain)

        self.sdr = sdr

        self.analysis_block_samples = self._finalize_analysis_block_samples()
        self._ring_capacity = int(self.sample_rate * self.sdr_timeout_s)
        if self._ring_capacity < self.analysis_block_samples:
            logger.critical(
                "SDR %s: ring capacity (%s samples, sdr_timeout_s=%s) is smaller than analysis_block_samples=%s. "
                "Increase sdr_timeout_s or lower fft_nperseg / analysis_block_samples.",
                self.device,
                self._ring_capacity,
                self.sdr_timeout_s,
                self.analysis_block_samples,
            )
            sys.exit(1)

        logger.info(
            "SDR %s: analysis_block_samples=%s (~%.3f s), ring_capacity=%s (~%.3f s), sdr_callback_length=%s",
            self.device,
            self.analysis_block_samples,
            self.analysis_block_samples / self.sample_rate,
            self._ring_capacity,
            self._ring_capacity / self.sample_rate,
            self.sdr_callback_length,
        )

        self._ring_buffer = np.empty(self._ring_capacity, dtype=np.complex64)
        self._ring_read = 0
        self._ring_write = 0
        self._ring_count = 0
        self._ring_cv = threading.Condition()
        self._consumer_stop = threading.Event()
        self._sample_clock_t0 = None
        self._samples_consumed = 0

        self._consumer_thread = threading.Thread(target=self._analysis_consumer_loop, name=f"SDR{self.device}-analyze")
        self._consumer_thread.start()

        signal.signal(signal.SIGALRM, self.handle_signal)
        signal.alarm(self.sdr_timeout_s)

        try:
            self.sdr.read_samples_async(self._producer_enqueue, self.sdr_callback_length)
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

        self.update_state(datetime.datetime.now(), StateMessage.State.STOPPED)
        if self._consumer_stop is not None:
            self._consumer_stop.set()
            if self._ring_cv is not None:
                with self._ring_cv:
                    self._ring_cv.notify_all()
        if self.sdr is not None:
            self.sdr.cancel_read_async()

    def update_state(self, ts: datetime.datetime, state: StateMessage.State):
        # skip update if there is a state
        if self.last_state:
            # the state is different
            if self.last_state.state == state:
                # the state's timeout isn't over
                if self.last_state.ts + datetime.timedelta(seconds=self.state_update_s) >= ts.astimezone():
                    return

        self.last_state = StateMessage(self.device, ts.astimezone(), state)
        self.signal_queue.put(self.last_state)

    def consume_signal(self, signal: Signal):
        """
        Puts a detected signals into the signal queue.

        Parameters
        ----------
        signal: radiotracking.Signal
            The signal to put into the signal queue.
        """
        logger.debug(f"SDR {self.device} received {signal}")
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
        logger.debug(f"shadow list: {signals_status}")

        return [sig for sig, shadow in zip(signals, signals_status) if shadow is None]

    def extract_signals(
        self, freqs: np.ndarray, times: np.ndarray, spectrogram: np.ndarray, ts_start: datetime.datetime
    ) -> List[Signal]:
        """Extract plateaus from spectogram data.

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

        Returns
        -------
        List[radiotracking.Signal]
            List of signals extracted from spectogram.
        """
        signals = []

        if len(times) == 0:
            return signals

        signal_min_duration_num = self.signal_min_duration / (times[1] - times[0])

        # iterate over all frequencies
        for fi, fft in enumerate(spectrogram):
            # set freq_avg to None to allow lazy evaluation
            freq_avg = None
            freq = freqs[fi] + self.center_freq
            ti_skip = 0

            # jump over all power values in signal_min_duration_num distance
            for ti in range(0, len(fft), max(1, int(signal_min_duration_num))):
                # skip values already inspected during a signal
                if ti < ti_skip:
                    continue

                # check if power of signal over threshold
                if fft[ti] < self.signal_threshold:
                    continue

                # lazy computation for freq_avg
                if freq_avg is None:
                    freq_avg = np.mean(fft)

                # check if snr of sample is below threshold
                if fft[ti] / freq_avg < self.snr_threshold:
                    continue

                # loop down until threshold is undershot
                start = ti
                start_min = 0 if self._spectrogram_last is None else -len(self._spectrogram_last[0]) + 1
                while start > start_min:
                    if start < 0:
                        power = self._spectrogram_last[fi, start]
                    else:
                        power = fft[start]

                    # check if power of signal over threshold
                    if power < self.signal_threshold:
                        break

                    # check if snr of sample is below threshold
                    if power / freq_avg < self.snr_threshold:
                        break

                    start -= 1

                # loop up until threshold is undershot
                end = ti
                while end < len(fft):
                    if fft[end] < self.signal_threshold:
                        ti_skip = end
                        break

                    # check if snr of sample is below threshold
                    if fft[end] / freq_avg < self.snr_threshold:
                        ti_skip = end
                        break

                    end += 1

                # skip signal, if it laps into next spectogram
                if end == len(fft):
                    logger.debug("signal overlaps to next spectogram, skipping")
                    continue

                # compute duration and skip, if too short
                end_dt = times[end]
                # if start has negative index
                if start < 0:
                    start_dt = -times[-start]
                else:
                    start_dt = times[start]

                duration_s = end_dt - start_dt
                duration = datetime.timedelta(seconds=duration_s)
                if duration_s < self.signal_min_duration:
                    continue
                if duration_s > self.signal_max_duration:
                    logger.debug(
                        f"signal duration too long ({duration_s * 1000} > {self.signal_max_duration*1000} ms), skipping"
                    )
                    continue
                ts = ts_start + datetime.timedelta(seconds=start_dt)

                # extract data
                if start < 0:
                    data = np.concatenate((self._spectrogram_last[fi][start:], fft[:end]))
                else:
                    data = fft[start:end]

                max_dBW = dB(np.max(data)) - self.calibration_db
                avg = np.mean(data)
                avg_dBW = dB(avg) - self.calibration_db
                std_dB = np.std(dB(data))
                noise_dBW = dB(freq_avg)
                snr_dB = dB(avg / freq_avg)

                signal = Signal(
                    self.device, ts.astimezone(), freq, duration, max_dBW, avg_dBW, std_dB, noise_dBW, snr_dB
                )
                signals.append(signal)

        return signals
