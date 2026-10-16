"""
The signal extraction of radiotracking before it was vectorized (per frequency, in Python), kept as reference.

The vectorized ``SignalAnalyzer.extract_signals`` must return exactly the same signals, see ``test_extract_equivalence.py``.
``self`` is a ``SignalAnalyzer``.
"""

import datetime
import logging
from typing import List, Optional

import numpy as np

from radiotracking import Signal, dB
from radiotracking.analyze import noise_levels

logger = logging.getLogger(__name__)


def reference_extract_signals(
    self,
    freqs: np.ndarray,
    times: np.ndarray,
    spectrogram: np.ndarray,
    ts_start: datetime.datetime,
    noise: Optional[np.ndarray] = None,
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
    noise: np.ndarray | None
        noise power of every frequency, see :func:`noise_levels`; computed if None

    Returns
    -------
    List[radiotracking.Signal]
        List of signals extracted from spectogram.
    """
    signals = []

    if len(times) == 0:
        return signals

    if noise is None:
        noise = noise_levels(spectrogram)

    # duration of one spectrogram segment (no overlap), known from the configuration, a block may hold a single one
    segment_duration = self.fft_nperseg / self.sample_rate
    signal_min_duration_num = self.signal_min_duration / segment_duration

    # segments of the previous block (negative indexes) a signal may start in
    last = self._spectrogram_last
    start_min = 0 if last is None else -len(last[0])

    # iterate over all frequencies
    for fi, fft in enumerate(spectrogram):
        # noise power of this frequency
        freq_avg = noise[fi]
        freq = freqs[fi] + self.center_freq
        ti_skip = start_min

        def power_at(i: int) -> float:
            return last[fi, i] if i < 0 else fft[i]

        # jump over all power values in signal_min_duration_num distance, starting with the last segment of the
        # previous block: a signal ending there was deferred to this block, as it might have continued
        seeds = range(0, len(fft), max(1, int(signal_min_duration_num)))
        for ti in ([-1] if last is not None else []) + list(seeds):
            # skip values already inspected during a signal
            if ti < ti_skip:
                continue

            # check if power of signal over threshold
            if power_at(ti) < self.signal_threshold:
                continue

            # check if snr of sample is below threshold
            if power_at(ti) / freq_avg < self.snr_threshold:
                continue

            # loop down until threshold is undershot, start is the first segment of the signal
            start = ti
            while start > start_min:
                power = power_at(start - 1)

                # check if power of signal over threshold
                if power < self.signal_threshold:
                    break

                # check if snr of sample is below threshold
                if power / freq_avg < self.snr_threshold:
                    break

                start -= 1

            # loop up until threshold is undershot, end is the first segment after the signal
            end = ti
            while end < len(fft):
                power = power_at(end)
                if power < self.signal_threshold:
                    break

                # check if snr of sample is below threshold
                if power / freq_avg < self.snr_threshold:
                    break

                end += 1
            ti_skip = end

            # skip signal, if it laps into next spectogram
            if end == len(fft):
                logger.debug("signal overlaps to next spectogram, skipping")
                continue

            # compute duration and skip, if too short; the signal spans the segments start .. end - 1, times are
            # taken from the segment edges (scipy's times are the segment centers)
            start_dt = start * segment_duration
            end_dt = end * segment_duration

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
                data = np.concatenate((last[fi][start:], fft[:end]))
            else:
                data = fft[start:end]

            max_dBW = dB(np.max(data)) - self.calibration_db
            avg = np.mean(data)
            avg_dBW = dB(avg) - self.calibration_db
            std_dB = np.std(dB(data))
            noise_dBW = dB(freq_avg) - self.calibration_db
            snr_dB = dB(avg / freq_avg)

            signal = Signal(
                self.device, ts.astimezone(), freq, duration, max_dBW, avg_dBW, std_dB, noise_dBW, snr_dB
            )
            signals.append(signal)

    return signals
