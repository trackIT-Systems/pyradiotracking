import csv
import datetime
import io
import json
import logging
import os
import socket
import sys
from importlib.metadata import version
from unittest import mock

import cbor2
import pytest

from radiotracking import MatchingSignal, Signal, StateMessage
from radiotracking.consume import (
    CSVConsumer,
    MQTTConsumer,
    ProcessConnector,
    cborify,
    csvify,
    jsonify,
    uncborify,
)

from .conftest import T0


def matching(make_signal, devices=("0", "1")):
    msig = MatchingSignal(list(devices))
    msig.add_member(make_signal(device="0", avg_dbw=-50))
    return msig


CBOR2_MAJOR = int(version("cbor2").split(".")[0])
cbor2_hook = pytest.mark.xfail(
    CBOR2_MAJOR >= 6,
    strict=True,
    reason="cbor2 6 calls tag hooks as hook(tag, immutable) instead of hook(decoder, tag); pyproject allows cbor2 >= 5.6.3 unbounded",
)


class TestSerialization:
    def test_jsonify(self):
        assert jsonify(T0) == T0.isoformat()
        assert jsonify(datetime.timedelta(milliseconds=20)) == 0.02
        with pytest.raises(TypeError):
            jsonify(object())

    def test_csvify(self):
        assert csvify(datetime.timedelta(seconds=1.5)) == 1.5
        assert csvify("x") == "x"

    @cbor2_hook
    def test_cbor_roundtrip(self, make_signal):
        sig = make_signal()
        raw = cbor2.dumps(sig.as_list, default=cborify)
        decoded = cbor2.loads(raw, tag_hook=uncborify)
        assert decoded[1] == T0
        assert decoded[3] == sig.duration
        assert decoded[2] == sig.frequency

    def test_cbor_state_message_with_none_metrics(self):
        raw = cbor2.dumps(StateMessage("0", T0, StateMessage.State.RUNNING).as_list, default=cborify)
        assert cbor2.loads(raw, tag_hook=uncborify)[3:] == [None, None, None]

    def test_json_of_matching_signal_with_missing_device(self, make_signal):
        msig = matching(make_signal)
        data = json.loads(json.dumps(msig.as_dict, default=jsonify))
        assert data["0"] == -50
        assert data["1"] is None


class TestCSVConsumer:
    def test_writes_header_and_only_matching_class(self, make_signal):
        out = io.StringIO()
        consumer = CSVConsumer(out, Signal, header=Signal.header)
        consumer.add(make_signal())
        consumer.add(StateMessage("0", T0, StateMessage.State.RUNNING))
        rows = list(csv.reader(io.StringIO(out.getvalue()), delimiter=";"))
        assert rows[0] == Signal.header
        assert len(rows) == 2
        assert rows[1][0] == "0"
        assert float(rows[1][3]) == 0.02  # duration in seconds

    def test_matching_signal_row_has_empty_cell_for_missing_device(self, make_signal):
        out = io.StringIO()
        consumer = CSVConsumer(out, MatchingSignal, header=MatchingSignal(["0", "1"]).header)
        consumer.add(matching(make_signal))
        rows = list(csv.reader(io.StringIO(out.getvalue()), delimiter=";"))
        assert len(rows[0]) == len(rows[1])
        assert rows[1][-1] == ""


class TestMQTTConsumer:
    @pytest.fixture
    def client(self):
        with mock.patch("paho.mqtt.client.Client") as cls:
            yield cls.return_value

    @pytest.fixture
    def consumer(self, client):
        c = MQTTConsumer("localhost", 1883, 1, 60, 0, prefix="station/radiotracking")
        yield c
        c.shutdown()

    def topics(self, client):
        return [call.args[0] for call in client.publish.call_args_list]

    def test_connects_and_starts_loop(self, client, consumer):
        client.connect.assert_called_once_with("localhost", 1883, keepalive=60)
        client.loop_start.assert_called_once()

    def test_signal_is_published_in_three_formats(self, client, consumer, make_signal):
        consumer.add(make_signal(device="1"))
        assert self.topics(client) == [
            "station/radiotracking/device/1/json",
            "station/radiotracking/device/1/csv",
            "station/radiotracking/device/1/cbor",
        ]

    def test_message_topics(self, client, consumer, make_signal):
        consumer.add(matching(make_signal))
        consumer.add(StateMessage("0", T0, StateMessage.State.STARTED))
        topics = self.topics(client)
        assert "station/radiotracking/matched/json" in topics
        assert "station/radiotracking/state/cbor" in topics

    @cbor2_hook
    def test_payloads_decode(self, client, consumer, make_signal):
        consumer.add(make_signal())
        payloads = {c.args[0].rsplit("/", 1)[1]: c.args[1] for c in client.publish.call_args_list}
        assert json.loads(payloads["json"])["Device"] == "0"
        assert payloads["csv"].split(";")[0] == "0"
        assert cbor2.loads(payloads["cbor"], tag_hook=uncborify)[0] == "0"

    def test_unknown_type_is_skipped(self, client, consumer):
        consumer.add(object())  # type: ignore[arg-type]
        client.publish.assert_not_called()

    def test_publish_errors_do_not_propagate(self, client, consumer, make_signal):
        client.publish.side_effect = OSError("broker gone")
        consumer.add(make_signal())

    def test_no_publish_after_shutdown(self, client, consumer, make_signal):
        consumer.shutdown()
        consumer.shutdown()
        client.loop_stop.assert_called_once()
        consumer.add(make_signal())
        client.publish.assert_not_called()

    def test_forwards_log_records_as_csv(self, client, consumer):
        consumer.emit(logging.LogRecord("radiotracking.x", logging.WARNING, __file__, 1, "hello; world", None, None))
        topic, payload = client.publish.call_args.args[:2]
        assert topic == "station/radiotracking/log/csv"
        assert next(csv.reader([payload], delimiter=";")) == ["WARNING", "radiotracking.x", "hello; world"]


class TestProcessConnector:
    def make(self, tmp_path, **over):
        kw = dict(
            station="st",
            device=["0", "1"],
            calibrate=False,
            sig_stdout=False,
            match_stdout=False,
            path=str(tmp_path),
            csv=False,
            mqtt=False,
        )
        kw.update(over)
        return ProcessConnector(**kw)

    def test_no_consumers_by_default(self, tmp_path):
        assert self.make(tmp_path).consumers == []

    def test_csv_files_are_created(self, tmp_path, make_signal):
        conn = self.make(tmp_path, csv=True)
        outdir = tmp_path / socket.gethostname() / "radiotracking"
        names = sorted(p.name for p in outdir.iterdir())
        assert len(names) == 3
        assert any(n.endswith("-matched.csv") for n in names)
        assert any(n.endswith("-state.csv") for n in names)

        conn.q.put(make_signal())
        conn.step(datetime.timedelta(seconds=2))
        signal_csv = next(p for p in outdir.iterdir() if not p.name.endswith(("-matched.csv", "-state.csv")))
        assert len(signal_csv.read_text().splitlines()) == 2

    def test_calibration_suffix(self, tmp_path):
        self.make(tmp_path, csv=True, calibrate=True)
        names = [p.name for p in (tmp_path / socket.gethostname() / "radiotracking").iterdir()]
        assert all(n.endswith("_calibration.csv") for n in names)

    def test_mqtt_is_skipped_in_calibration(self, tmp_path):
        with mock.patch("paho.mqtt.client.Client"):
            conn = self.make(tmp_path, mqtt=True, calibrate=True, mqtt_host="h", mqtt_port=1, mqtt_qos=1, mqtt_keepalive=1, mqtt_verbose=0)
        assert conn.consumers == []

    def test_mqtt_consumer_attaches_log_handler_and_shutdown_detaches(self, tmp_path):
        with mock.patch("paho.mqtt.client.Client"):
            conn = self.make(tmp_path, mqtt=True, mqtt_host="h", mqtt_port=1, mqtt_qos=1, mqtt_keepalive=1, mqtt_verbose=0)
        handler = conn.consumers[0]
        assert isinstance(handler, MQTTConsumer)
        assert handler in logging.getLogger("radiotracking").handlers
        conn.shutdown()
        conn.shutdown()
        assert handler not in logging.getLogger("radiotracking").handlers

    def test_step_returns_on_timeout(self, tmp_path):
        self.make(tmp_path).step(datetime.timedelta(seconds=0.05))

    def test_step_dispatches_to_all_consumers(self, tmp_path, make_signal):
        conn = self.make(tmp_path)
        seen = []
        conn.consumers.extend([mock.Mock(add=seen.append), mock.Mock(add=seen.append)])
        sig = make_signal()
        conn.q.put(sig)
        conn.step(datetime.timedelta(seconds=2))
        # signals pass through a multiprocessing queue and arrive as copies
        assert [repr(x) for x in seen] == [repr(sig), repr(sig)]

    def test_step_survives_negative_timeout(self, tmp_path):
        # main loop passes `next_check - now`, which can already be negative
        self.make(tmp_path).step(datetime.timedelta(seconds=-1))

    @pytest.mark.xfail(strict=True, reason="a failing consumer aborts the step, so the remaining consumers never see the message")
    def test_failing_consumer_does_not_starve_others(self, tmp_path, make_signal):
        conn = self.make(tmp_path)
        seen = []
        conn.consumers.extend([mock.Mock(add=mock.Mock(side_effect=RuntimeError("boom"))), mock.Mock(add=seen.append)])
        conn.q.put(make_signal())
        conn.step(datetime.timedelta(seconds=2))
        assert len(seen) == 1
