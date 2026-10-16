import datetime
import logging
import statistics
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

__version__ = "0.4.0"

logger = logging.getLogger(__name__)


def dB(val: float):
    """
    Convert a power value to dB.
    """
    return 10 * np.log10(val)


def from_dB(dB: float):
    """Convert a dB value to power."""
    return 10 ** (dB / 10)


class AbstractMessage(ABC):
    """
    Abstract class for a message.
    """

    header: List[str]
    """Header for the as_list() method."""

    def __init__(self) -> None:
        super().__init__()

        self.ts: datetime.datetime
        """Timestamp of the message."""

    @property
    @abstractmethod
    def as_list(self) -> List:
        """
        Return the message as a list of values.

        Returns
        -------
        typing.List[typing.Any]
        """

    @property
    def as_dict(self) -> Dict:
        """
        Return the message as a dictionary.

        Returns
        -------
        typing.Dict[str, typing.Any]
        """
        return dict(zip(self.header, self.as_list))


class StateMessage(AbstractMessage):
    class State(Enum):
        STOPPED = 0
        RUNNING = 1
        STARTED = 2

    def __init__(
        self,
        device: str,
        ts: datetime.datetime,
        state: Union["StateMessage.State", int, str],
    ):
        super().__init__()

        self.device: str = device
        self.ts: datetime.datetime = ts
        self.state: StateMessage.State
        if isinstance(state, StateMessage.State):
            self.state = state
        else:
            self.state = StateMessage.State(int(state))

    header: List[str] = ["Device", "Time", "State"]

    @property
    def as_list(self) -> List:
        return [
            self.device,
            self.ts,
            self.state.value,
        ]

    def __repr__(self) -> str:
        return f"StateMessage({self.device}, {self.ts}, {self.state})"


class SpectrumMessage(AbstractMessage):
    """
    Noise and SNR of every frequency bin of a device over an interval, independent of detected signals.

    Parameters
    ----------
    device: str
        The device the spectrum was measured on.
    ts: typing.Union[datetime.datetime, str]
        The start of the interval.
    duration: typing.Union[datetime.timedelta, float, str]
        The duration of IQ samples the interval covers.
    frequency: typing.Union[float, str]
        The frequency of the lowest bin (Hz).
    bin_width: typing.Union[float, str]
        The width of a bin (Hz), the bins are in ascending frequency.
    noise: typing.Sequence[typing.Union[float, str]]
        The noise level of every bin (dBW).
    snr: typing.Sequence[typing.Union[float, str]]
        The ratio of the strongest power in every bin to its noise level (dB).
    """

    def __init__(
        self,
        device: str,
        ts: Union[datetime.datetime, str],
        duration: Union[datetime.timedelta, float, str],
        frequency: Union[float, str],
        bin_width: Union[float, str],
        noise: Sequence[Union[float, str]],
        snr: Sequence[Union[float, str]],
    ):
        super().__init__()

        self.device = device
        if isinstance(ts, datetime.datetime):
            self.ts = ts
        else:
            self.ts = datetime.datetime.fromisoformat(ts)
        if isinstance(duration, datetime.timedelta):
            self.duration = duration
        else:
            self.duration = datetime.timedelta(seconds=float(duration))
        self.frequency = float(frequency)
        self.bin_width = float(bin_width)
        self.noise: List[float] = [float(n) for n in noise]
        self.snr: List[float] = [float(n) for n in snr]
        if len(self.noise) != len(self.snr):
            raise ValueError(f"noise ({len(self.noise)}) and snr ({len(self.snr)}) must have one value per bin")

    header: List[str] = ["Device", "Time", "Duration", "Frequency (Hz)", "Bin Width (Hz)", "Noise (dBW)", "SNR (dB)"]

    @property
    def frequencies(self) -> List[float]:
        """The frequency of every bin (Hz)."""
        return [self.frequency + i * self.bin_width for i in range(len(self.noise))]

    @property
    def as_list(self) -> List:
        # half precision resolves 1/16 dB at -100 dBW, and halves the size of the values in CBOR
        return [
            self.device,
            self.ts,
            self.duration,
            self.frequency,
            self.bin_width,
            [float(np.float16(n)) for n in self.noise],
            [float(np.float16(n)) for n in self.snr],
        ]

    def __repr__(self) -> str:
        return (
            f"SpectrumMessage({self.device}, {self.ts}, {self.duration}, {self.frequency}, {self.bin_width}, "
            f"{len(self.noise)} bins)"
        )


class AbstractSignal(AbstractMessage):
    """
    Abstract class for a signal.
    """

    def __init__(self) -> None:
        super().__init__()

        self.frequency: float
        """Frequency of the signal in Hz."""
        self.duration: datetime.timedelta
        """Duration of the signal."""


class Signal(AbstractSignal):
    """
    Signal detected on a single device.

    Parameters
    ----------
    device: str
        The device that detected the signal.
    ts: typing.Union[datetime.datetime, str]
        The timestamp of the signal.
    frequency: typing.Union[float, str]
        The frequency of the signal.
    duration: typing.Union[datetime.timedelta, float, str]
        The duration of the signal.
    max_dBW: typing.Union[float, str]
        The maximum power of the signal.
    avg_dBW: typing.Union[float, str]
        The average power of the signal.
    std_dB: typing.Union[float, str]
        The standard deviation in power of the signal.
    noise_dBW: typing.Union[float, str]
        The noise level of the signal.
    snr_dB: typing.Union[float, str]
        The signal to noise ratio of the signal.
    """

    def __init__(
        self,
        device: str,
        ts: Union[datetime.datetime, str],
        frequency: Union[float, str],
        duration: Union[datetime.timedelta, float, str],
        max_dBW: Union[float, str],
        avg_dBW: Union[float, str],
        std_dB: Union[float, str],
        noise_dBW: Union[float, str],
        snr_dB: Union[float, str],
    ):
        super().__init__()

        self.device = device
        if isinstance(ts, datetime.datetime):
            self.ts = ts
        else:
            self.ts = datetime.datetime.fromisoformat(ts)
        self.frequency = float(frequency)
        if isinstance(duration, datetime.timedelta):
            self.duration = duration
        else:
            self.duration = datetime.timedelta(seconds=float(duration))

        self.max: float = float(max_dBW)
        """The maximum power of the signal."""
        self.avg: float = float(avg_dBW)
        """The average power of the signal."""
        self.std: float = float(std_dB)
        """The standard deviation in power of the signal."""
        self.noise: float = float(noise_dBW)
        """The noise level of the signal."""
        self.snr: float = float(snr_dB)
        """The signal to noise ratio of the signal."""

    header = [
        "Device",
        "Time",
        "Frequency",
        "Duration",
        "max (dBW)",
        "avg (dBW)",
        "std (dB)",
        "noise (dBW)",
        "snr (dB)",
    ]

    @property
    def as_list(self):
        return [
            self.device,
            self.ts,
            self.frequency,
            self.duration,
            self.max,
            self.avg,
            self.std,
            self.noise,
            self.snr,
        ]

    def __repr__(self):
        return f"Signal({self.device}, {self.ts}, {self.frequency}, {self.duration}, {self.max}, {self.avg}, {self.std}, {self.noise}, {self.snr})"

    def __str__(self):
        return f"Signal<SDR {self.device}, {self.frequency/1000/1000:.3f} MHz, {self.duration.total_seconds()*1000:.2f} ms, {self.max:.1f} dBW>"


class MatchedSignal(AbstractSignal):
    """
    Matched Signal detected on multiple devices.

    Parameters
    ----------
    devices: typing.List[str]
        The devices that could have detected the signal.
    ts: typing.Union[datetime.datetime, str]
        The timestamp of the signal.
    frequency: typing.Union[float, str]
        The frequency of the signal.
    duration: typing.Union[datetime.timedelta, float, str]
        The duration of the signal.
    avgs: typing.List[float]
        The average powers detected on the available devices.
    """

    def __init__(
        self,
        devices: List[str],
        ts: Union[datetime.datetime, str],
        frequency: Union[float, str],
        duration: Union[datetime.timedelta, float, str],
        *avgs: float,
    ):
        super().__init__()
        self.devices = devices

        if isinstance(ts, datetime.datetime):
            self.ts = ts
        else:
            self.ts = datetime.datetime.fromisoformat(ts)
        self.frequency = float(frequency)
        if isinstance(duration, datetime.timedelta):
            self.duration = duration
        else:
            self.duration = datetime.timedelta(seconds=float(duration))

        self._avgs: List[float | None] = []
        for avg in avgs:
            try:
                self._avgs.append(float(avg))
            except TypeError:
                self._avgs.append(None)

    @property
    def header(self) -> List[str]:
        """Header for the as_list() method."""
        return [
            "Time",
            "Frequency",
            "Duration",
            *self.devices,
        ]

    @property
    def as_list(self) -> List:
        return [self.ts, self.frequency, self.duration, *self._avgs]

    def __repr__(self) -> str:
        avgs_str = ", ".join([repr(avg) for avg in self._avgs])
        return f"MatchedSignal({self.devices}, {self.ts}, {self.frequency}, {self.duration}, {avgs_str})"

    def __str__(self):
        avgs_str = ", ".join([f"{avg:.2f}" if avg is not None else f"{None}" for avg in self._avgs])
        return f"{self.__class__.__name__}<SDRs {self.devices}, {self.frequency/1000/1000:.3f} MHz, {self.duration.total_seconds()*1000:.2f} ms, dBWs: [{avgs_str}]>"


class MatchingSignal(MatchedSignal):
    """
    Class for matching signals detected on multiple devices.

    Parameters
    ----------
    devices: typing.List[str]
        The devices that are available to detect signals.
    """

    def __init__(self, devices: List[str]):
        self.devices = devices
        self._sigs: Dict[str, Signal] = {}
        self._created: float = 0.0
        """Monotonic time the first member arrived, used by the matcher to flush on timeout."""

    @property
    def duration(self) -> datetime.timedelta:
        """
        Duration of the matching signal based on the detected maximum.

        Returns
        -------
        datetime.timedelta
        """
        return max([sig.duration for sig in self._sigs.values()])

    @property
    def ts(self) -> datetime.datetime:
        """
        Timestamp of the matching signal based on the earliest detection.

        Returns
        -------
        datetime.datetime
        """
        return min([sig.ts for sig in self._sigs.values()])

    @property
    def frequency(self) -> float:
        """
        Frequency of the matching signal based on median frequency.

        Returns
        -------
        float
        """
        return statistics.median([sig.frequency for sig in self._sigs.values()])

    @property
    def _avgs(self) -> List[float | None]:
        """
        Average powers of the matching signal.

        Returns
        -------
        typing.List[float]
        """
        return [self._sigs[d].avg if d in self._sigs else None for d in self.devices]

    def has_member(
        self,
        sig: Signal,
        time_diff: datetime.timedelta = datetime.timedelta(seconds=0),
        bandwidth: float = 0,
        duration_diff: Optional[datetime.timedelta] = None,
    ) -> bool:
        """
        Checks if a Signal is part of this matching signal.

        Parameters
        ----------
        sig: radiotracking.Signal
            The signal to check.
        time_diff: datetime.timedelta
            Allowed difference of the timestamp.
        bandwidth: float
            Allowed difference of the frequency.
        duration_diff: datetime.timedelta
            Allowed difference of the duration.

        Returns
        -------
        bool
            True if the signal is part of this matching signal.
        """

        # if freq (including bw) out of range of freq
        if sig.frequency - bandwidth / 2 > self.frequency:
            logger.debug(f"1 {sig.frequency - bandwidth / 2} > {self.frequency}")
            return False
        if sig.frequency + bandwidth / 2 < self.frequency:
            logger.debug(f"2 {sig.frequency + bandwidth / 2} < {self.frequency}")
            return False

        # if start (minus diff) is after end
        if sig.ts - time_diff > (self.ts + self.duration):
            logger.debug(f"3 {sig.ts - time_diff} > {(self.ts + self.duration)}")
            return False
        # if end (plus diff) is before start
        if (sig.ts + sig.duration) + time_diff < self.ts:
            logger.debug(f"4 {(sig.ts + sig.duration) + time_diff} < {self.ts}")
            return False

        # if no duration_diff is present, don't match for it
        if duration_diff:
            if sig.duration - (duration_diff / 2) > self.duration:
                return False
            if sig.duration + (duration_diff / 2) < self.duration:
                return False

        return True

    def add_member(self, sig: Signal):
        """
        Adds a signal to the matching signal.

        Parameters
        ----------
        sig: radiotracking.Signal
            The signal to add.
        """
        if sig.device in self._sigs:
            logger.info(f"{sig} already contained in {self}")
            if self._sigs[sig.device].avg < sig.avg:
                logger.info(f"Replacing initial {self._sigs[sig.device]}")
                self._sigs[sig.device] = sig
            else:
                logger.info(f"Keeping initial {self._sigs[sig.device]}")
        else:
            self._sigs[sig.device] = sig
