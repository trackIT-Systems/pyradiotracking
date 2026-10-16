import csv
import datetime
import json
import logging
import multiprocessing
import os
import platform
import queue
import socket
import sys
from abc import ABC, abstractmethod
from io import StringIO
from typing import List, Optional, Type

import cbor2 as cbor
import paho.mqtt.client

from radiotracking import AbstractMessage, MatchingSignal, Signal, SpectrumMessage, StateMessage

logger = logging.getLogger(__name__)

MQTT_MAX_QUEUED_MESSAGES = 10_000
"""Messages paho keeps while the broker is unreachable (about 3 per signal), see :class:`MQTTConsumer`."""


def jsonify(o):
    """Helper function to convert non-native types to JSON serializable values."""
    if isinstance(o, datetime.datetime):
        return o.isoformat()
    if isinstance(o, datetime.timedelta):
        return o.total_seconds()

    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def cborify(encoder, o):
    """Helper function to convert non-native types to CBOR serializable values."""
    if isinstance(o, datetime.timedelta):
        encoder.encode(cbor.CBORTag(1337, o.total_seconds()))


def uncborify(*args, **kwargs):
    """Helper function to convert CBOR tags to their original datatype.

    Used as ``tag_hook`` of cbor2, which calls it as ``hook(decoder, tag)`` up to version 5 and
    as ``hook(tag, immutable)`` from version 6 on, so the tag is looked up in the arguments.
    """
    tag = next(arg for arg in args if isinstance(arg, cbor.CBORTag))
    if tag.tag == 1337:
        return datetime.timedelta(seconds=tag.value)

    return tag


def csvify(o):
    """Helper function to convert non-native types to CSV serializable values."""
    if isinstance(o, datetime.timedelta):
        return o.total_seconds()
    if isinstance(o, (list, tuple)):
        return ",".join(f"{v:.2f}" for v in o)

    return o


class AbstractConsumer(ABC):
    """Abstract base class for consumers."""

    @abstractmethod
    def add(self, signal: AbstractMessage):
        """Add a signal to the consumer."""
        pass


class MQTTConsumer(AbstractConsumer):
    """Class implementing a consumer that publishes data to an MQTT broker.

    Parameters
    ----------
    mqtt_host : str
        The hostname of the MQTT broker.
    mqtt_port: int
        The port of the MQTT broker.
    mqtt_qos: int
        The quality of service to use.
    mqtt_keepalive: int
        The keepalive interval in seconds.
    prefix: str
        The prefix to use for the MQTT topics.
    """

    def __init__(
        self,
        mqtt_host: str,
        mqtt_port: int,
        mqtt_qos: int,
        mqtt_keepalive: int,
        prefix: str = "/radiotracking",
        **kwargs,
    ):
        self.prefix = prefix
        self.mqtt_qos = mqtt_qos
        self._shutdown = False
        # Publishing does not wait for the broker, paho's network thread sends the queued messages. Only on shutdown
        # the last message is waited for, bounded so a hung broker cannot block the clean exit.
        self._shutdown_timeout_s = 2.0
        self._last_publish: Optional[paho.mqtt.client.MQTTMessageInfo] = None
        self.client = paho.mqtt.client.Client(
            callback_api_version=paho.mqtt.client.CallbackAPIVersion.VERSION2,
            client_id=f"{platform.node()}-radiotracking",
            clean_session=False,
        )
        # bounds the memory while the broker is unreachable, newer messages are dropped beyond it
        self.client.max_queued_messages_set(MQTT_MAX_QUEUED_MESSAGES)
        self.client.connect(mqtt_host, mqtt_port, keepalive=mqtt_keepalive)
        self.client.loop_start()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def shutdown(self) -> None:
        """Stop the MQTT client. Safe to call multiple times."""
        if self._shutdown:
            return
        self._shutdown = True
        try:
            if self._last_publish is not None:
                self._last_publish.wait_for_publish(timeout=self._shutdown_timeout_s)
        except Exception:
            pass
        try:
            self.client.loop_stop()
        except Exception:
            pass
        try:
            self.client.disconnect()
        except Exception:
            pass

    def _publish(self, topic: str, payload, qos: int | None = None) -> None:
        if self._shutdown:
            return
        qos = self.mqtt_qos if qos is None else qos
        try:
            self._last_publish = self.client.publish(topic, payload, qos=qos)
        except Exception:
            return

    def add(self, signal: AbstractMessage):
        """Add a signal to the consumer."""
        if self._shutdown:
            return

        if isinstance(signal, Signal):
            path = f"{self.prefix}/device/{signal.device}"
        elif isinstance(signal, MatchingSignal):
            path = f"{self.prefix}/matched"
        elif isinstance(signal, StateMessage):
            path = f"{self.prefix}/state"
        elif isinstance(signal, SpectrumMessage):
            # hundreds of values per message: only compact CBOR, the half precision floats take 3 bytes each
            payload_cbor = cbor.dumps(signal.as_list, default=cborify, canonical=True)
            self._publish(f"{self.prefix}/spectrum/cbor", payload_cbor)
            logger.debug(f"published spectrum via mqtt, cbor: {len(payload_cbor)}")
            return
        else:
            logger.critical(f"Unknown data type {type(signal)}, skipping.")
            return

        # publish json
        payload_json = json.dumps(
            signal.as_dict,
            default=jsonify,
        )
        self._publish(path + "/json", payload_json)

        # publish csv
        csv_io = StringIO()
        csv.writer(csv_io, dialect="excel", delimiter=";").writerow([csvify(v) for v in signal.as_list])
        payload_csv = csv_io.getvalue().splitlines()[0]
        self._publish(path + "/csv", payload_csv)

        # publish cbor
        payload_cbor = cbor.dumps(
            signal.as_list,
            default=cborify,
        )
        self._publish(path + "/cbor", payload_cbor)

        logger.debug(
            f"published via mqtt, json: {len(payload_json)}, csv: {len(payload_csv)}, cbor: {len(payload_cbor)}"
        )


class CSVConsumer(AbstractConsumer):
    """
    Class implementing a consumer that writes a local CSV file.

    Parameters
    ----------
    filename : str
        The filename of the CSV file.
    cls : Type[AbstractMessage]
        The type of signals to be written to the CSV file.
    header : List[str]
        The header of the CSV file.
    """

    def __init__(
        self,
        out,
        cls: Type[AbstractMessage],
        header: List[str] | None = None,
    ):
        self.out = out
        self.cls = cls

        self.writer = csv.writer(out, dialect="excel", delimiter=";")
        if header:
            self.writer.writerow(header)
        self.out.flush()

    def add(self, signal: AbstractMessage):
        """Add a signal to the consumer."""
        if isinstance(signal, self.cls):
            self.writer.writerow([csvify(v) for v in signal.as_list])
            self.out.flush()

            logger.debug(f"published {signal} via csv")
        else:
            pass


class ProcessConnector:
    """
    Connects the running analysis processes to the consumers.

    Parameters
    ----------
    stations : List[str]
        Name of the station.
    device : List[str]
        Name of the devices used by the station.
    calibrate : bool
        Whether to use the signals to calibrate the station.
    sig_stdout : bool
        Whether to write detected signals to stdout.
    match_stdout : bool
        Whether to write matched signals to stdout.
    path : str
        The path to write output files.
    csv : bool
        Whether to write CSV files.
    mqtt : bool
        Whether to publish data via MQTT.
    """

    def __init__(
        self,
        station: str,
        device: List[str],
        calibrate: bool,
        sig_stdout: bool,
        match_stdout: bool,
        path: str,
        csv: bool,
        mqtt: bool,
        **kwargs,
    ):
        self.q: multiprocessing.Queue[AbstractMessage] = multiprocessing.Queue()
        self.consumers: List[AbstractConsumer] = []
        """List of consumers data is published to."""

        ts = datetime.datetime.now()

        # add stdout consumers
        if sig_stdout:
            sig_stdout_consumer = CSVConsumer(sys.stdout, Signal)
            self.consumers.append(sig_stdout_consumer)
        if match_stdout:
            match_stdout_consumer = CSVConsumer(sys.stdout, MatchingSignal)
            self.consumers.append(match_stdout_consumer)

        # add csv consumer
        if csv:
            path = f"{path}/{socket.gethostname()}/radiotracking"
            # create output directory
            os.makedirs(path, exist_ok=True)

            # create consumer for signals
            signal_csv_path = f"{path}/{station}_{ts:%Y-%m-%dT%H%M%S}"
            signal_csv_path += "_calibration" if calibrate else ""
            signal_csv_consumer = CSVConsumer(open(f"{signal_csv_path}.csv", "w"), cls=Signal, header=Signal.header)
            self.consumers.append(signal_csv_consumer)

            # create consumer for matched signals
            matched_csv_path = f"{path}/{station}_{ts:%Y-%m-%dT%H%M%S}-matched"
            matched_csv_path += "_calibration" if calibrate else ""
            matched_csv_consumer = CSVConsumer(
                open(f"{matched_csv_path}.csv", "w"), cls=MatchingSignal, header=MatchingSignal(device).header
            )
            self.consumers.append(matched_csv_consumer)

            # create consumer for state information
            state_csv_path = f"{path}/{station}_{ts:%Y-%m-%dT%H%M%S}-state"
            state_csv_path += "_calibration" if calibrate else ""
            state_csv_consumer = CSVConsumer(
                open(f"{state_csv_path}.csv", "w"), cls=StateMessage, header=StateMessage.header
            )
            self.consumers.append(state_csv_consumer)

            # create consumer for the noise and snr per frequency
            spectrum_csv_path = f"{path}/{station}_{ts:%Y-%m-%dT%H%M%S}-spectrum"
            spectrum_csv_path += "_calibration" if calibrate else ""
            spectrum_csv_consumer = CSVConsumer(
                open(f"{spectrum_csv_path}.csv", "w"), cls=SpectrumMessage, header=SpectrumMessage.header
            )
            self.consumers.append(spectrum_csv_consumer)

        # add mqtt consumer (only if not in calibration)
        if mqtt and not calibrate:
            mqtt_consumer = MQTTConsumer(prefix=f"{station}/radiotracking", **kwargs)
            self.consumers.append(mqtt_consumer)

        self._connector_shutdown = False

    def shutdown(self) -> None:
        """Stop MQTT so shutdown cannot block on the broker."""
        if getattr(self, "_connector_shutdown", False):
            return
        self._connector_shutdown = True
        for c in self.consumers:
            if isinstance(c, MQTTConsumer):
                c.shutdown()

    def step(self, timeout: datetime.timedelta) -> bool:
        """
        Method that waits for new signals and publishes them to the consumers, blocking.

        Parameters
        ----------
        timeout : datetime.timedelta
            The timeout for the blocking call.

        Returns
        -------
        bool
            True if a message was published, False if the queue was empty for the whole timeout."""
        try:
            sig = self.q.get(timeout=timeout.total_seconds())
        except queue.Empty:
            return False

        for c in self.consumers:
            try:
                c.add(sig)
            except Exception:
                # one failing consumer must not keep the others from seeing the message
                logger.exception(f"{type(c).__name__} failed to consume {sig}")
        return True
