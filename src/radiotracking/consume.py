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
from typing import List, Type

import cbor2 as cbor
import paho.mqtt.client

from radiotracking import AbstractMessage, MatchingSignal, Signal, StateMessage

logger = logging.getLogger(__name__)


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


def uncborify(decoder, tag, shareable_index=None):
    """Helper function to convert CBOR tags to their original datatype."""
    if tag.tag == 1337:
        return datetime.timedelta(seconds=tag.value)

    return tag


def csvify(o):
    """Helper function to convert non-native types to CSV serializable values."""
    if isinstance(o, datetime.timedelta):
        return o.total_seconds()

    return o


class AbstractConsumer(ABC):
    """Abstract base class for consumers."""

    @abstractmethod
    def add(self, signal: AbstractMessage):
        """Add a signal to the consumer."""
        pass


class MQTTConsumer(logging.StreamHandler, AbstractConsumer):
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
    mqtt_verbose: int
        The verbosity level for log messages to be forwarded.
    prefix: str
        The prefix to use for the MQTT topics.
    """

    def __init__(
        self,
        mqtt_host: str,
        mqtt_port: int,
        mqtt_qos: int,
        mqtt_keepalive: int,
        mqtt_verbose: int,
        prefix: str = "/radiotracking",
        **kwargs,
    ):
        logging_level = max(0, logging.WARN - (mqtt_verbose * 10))
        logging.StreamHandler.__init__(self)
        super(logging.StreamHandler, self).__init__(level=logging_level)

        fmt = logging.Formatter("%(message)s")
        self.setFormatter(fmt)

        self.prefix = prefix
        self.mqtt_qos = mqtt_qos
        self.client = paho.mqtt.client.Client(f"{platform.node()}-radiotracking", clean_session=False)
        self.client.connect(mqtt_host, mqtt_port, keepalive=mqtt_keepalive)
        self.client.loop_start()

    def __del__(self):
        logger.info("Stopping MQTT thread")
        self.client.loop_stop()

    def emit(self, record):
        """Override the emit method to forward log messages to the MQTT broker."""
        path = f"{self.prefix}/log"

        # skip dash messages
        if record.name.startswith("radiotracking.present"):
            return

        # publish csv
        csv_io = StringIO()
        csv.writer(csv_io, dialect="excel", delimiter=";").writerow(
            [record.levelname, record.name, self.format(record)]
        )
        payload_csv = csv_io.getvalue().splitlines()[0]
        self.client.publish(path + "/csv", payload_csv, qos=self.mqtt_qos)

    def add(self, signal: AbstractMessage):
        """Add a signal to the consumer."""

        if isinstance(signal, Signal):
            path = f"{self.prefix}/device/{signal.device}"
        elif isinstance(signal, MatchingSignal):
            path = f"{self.prefix}/matched"
        elif isinstance(signal, StateMessage):
            path = f"{self.prefix}/state"
        else:
            logger.critical(f"Unknown data type {type(signal)}, skipping.")
            return

        # publish json
        payload_json = json.dumps(
            signal.as_dict,
            default=jsonify,
        )
        self.client.publish(path + "/json", payload_json, qos=self.mqtt_qos)

        # publish csv
        csv_io = StringIO()
        csv.writer(csv_io, dialect="excel", delimiter=";").writerow([csvify(v) for v in signal.as_list])
        payload_csv = csv_io.getvalue().splitlines()[0]
        self.client.publish(path + "/csv", payload_csv, qos=self.mqtt_qos)

        # publish cbor
        payload_cbor = cbor.dumps(
            signal.as_list,
            default=cborify,
        )
        self.client.publish(path + "/cbor", payload_cbor, qos=self.mqtt_qos)

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

        # add mqtt consumer (only if not in calibration)
        if mqtt and not calibrate:
            mqtt_consumer = MQTTConsumer(prefix=f"{station}/radiotracking", **kwargs)
            self.consumers.append(mqtt_consumer)
            logging.getLogger("radiotracking").addHandler(mqtt_consumer)

    def step(self, timeout: datetime.timedelta):
        """
        Method that waits for new signals and publishes them to the consumers, blocking.

        Parameters
        ----------
        timeout : datetime.timedelta
            The timeout for the blocking call."""
        try:
            sig = self.q.get(timeout=timeout.total_seconds())
        except queue.Empty:
            return

        [c.add(sig) for c in self.consumers]
