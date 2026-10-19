import csv
import datetime
import io
import json
import logging
import os
import socket
from unittest import mock

import cbor2
import pytest

from radiotracking import CalibratedMatchedSignal, GainMessage, MatchingSignal, Signal, SpectrumMessage, StateMessage
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


def spectrum(bins=256):
    return SpectrumMessage(
        "0",
        T0,
        datetime.timedelta(seconds=300),
        150_005_000.0,
        1171.875,
        # multiples of 1/16 dB, exact in half precision
        [-100.0 - (i % 7) * 0.125 for i in range(bins)],
        [3.0 + (i % 5) * 0.1875 for i in range(bins)],
    )


def matching(make_signal, devices=("0", "1")):
    msig = MatchingSignal(list(devices))
    msig.add_member(make_signal(device="0", avg_dbw=-50))
    return msig


class TestSerialization:
    def test_jsonify(self):
        assert jsonify(T0) == T0.isoformat()
        assert jsonify(datetime.timedelta(milliseconds=20)) == 0.02
        with pytest.raises(TypeError):
            jsonify(object())

    def test_csvify(self):
        assert csvify(datetime.timedelta(seconds=1.5)) == 1.5
        assert csvify("x") == "x"

    def test_cbor_roundtrip(self, make_signal):
        sig = make_signal()
        raw = cbor2.dumps(sig.as_list, default=cborify)
        decoded = cbor2.loads(raw, tag_hook=uncborify)
        assert decoded[1] == T0
        assert decoded[3] == sig.duration
        assert decoded[2] == sig.frequency

    def test_cbor_state_message_has_the_lifecycle_only(self):
        raw = cbor2.dumps(StateMessage("0", T0, StateMessage.State.RUNNING).as_list, default=cborify)
        assert cbor2.loads(raw, tag_hook=uncborify) == ["0", T0, 1]

    def test_csvify_joins_lists(self):
        assert csvify([-100.004, 3.5]) == "-100.00,3.50"

    def test_json_of_matching_signal_with_missing_device(self, make_signal):
        msig = matching(make_signal)
        data = json.loads(json.dumps(msig.as_dict, default=jsonify))
        assert data["0"] == -50
        assert data["1"] is None


class TestCSVConsumer:
    def test_spectrum_row_has_the_values_in_two_cells(self):
        out = io.StringIO()
        CSVConsumer(out, SpectrumMessage, header=SpectrumMessage.header).add(spectrum(bins=3))
        header, row = list(csv.reader(io.StringIO(out.getvalue()), delimiter=";"))
        assert header == SpectrumMessage.header
        assert row[2:5] == ["300.0", "150005000.0", "1171.875"]
        assert row[5] == "-100.00,-100.12,-100.25"
        assert row[6] == "3.00,3.19,3.38"

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
        c = MQTTConsumer("localhost", 1883, 1, 60, prefix="station/radiotracking")
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

    def test_calibrated_match_and_gain_topics(self, client, consumer):
        consumer.add(CalibratedMatchedSignal(["0", "1"], T0, 150_100_000.0, 0.02, -100.0, None))
        consumer.add(GainMessage("1", T0, 64.5, 15, 14, 7, 0.0, 32.03, -20.0, 76.5))
        topics = self.topics(client)
        assert "station/radiotracking/calibrated/cbor" in topics
        assert "station/radiotracking/gain/1/json" in topics
        assert "station/radiotracking/gain/1/cbor" in topics

    def test_gain_is_retained_and_signals_are_not(self, client, consumer, make_signal):
        consumer.add(GainMessage("1", T0, None, 15, 14, 7, 0.0, 32.03, None, None))
        consumer.add(make_signal())
        retained = {c.args[0]: c.kwargs["retain"] for c in client.publish.call_args_list}
        assert all(v for t, v in retained.items() if "/gain/" in t)
        assert not any(v for t, v in retained.items() if "/device/" in t)

    def test_calibrated_signal_publishes_the_appended_values(self, client, consumer):
        sig = Signal("0", T0, 150_100_000.0, 0.02, -50, -55, 1.0, -80, 25, -90.0, -95.0, -115.0)
        consumer.add(sig)
        payloads = {c.args[0].rsplit("/", 1)[1]: c.args[1] for c in client.publish.call_args_list}
        assert json.loads(payloads["json"])["max (dBm)"] == -90.0
        assert cbor2.loads(payloads["cbor"], tag_hook=uncborify)[9:] == [-90.0, -95.0, -115.0]
        assert payloads["csv"].split(";")[9:] == ["-90.0", "-95.0", "-115.0"]

    def test_payloads_decode(self, client, consumer, make_signal):
        consumer.add(make_signal())
        payloads = {c.args[0].rsplit("/", 1)[1]: c.args[1] for c in client.publish.call_args_list}
        assert json.loads(payloads["json"])["Device"] == "0"
        assert payloads["csv"].split(";")[0] == "0"
        assert cbor2.loads(payloads["cbor"], tag_hook=uncborify)[0] == "0"

    def test_spectrum_is_published_in_three_formats(self, client, consumer):
        consumer.add(spectrum())
        assert self.topics(client) == [
            "station/radiotracking/spectrum/json",
            "station/radiotracking/spectrum/csv",
            "station/radiotracking/spectrum/cbor",
        ]
        payloads = {c.args[0].rsplit("/", 1)[1]: c.args[1] for c in client.publish.call_args_list}

        data = json.loads(payloads["json"])
        assert data["Device"] == "0"
        assert data["Noise (dBW)"] == spectrum().noise
        assert data["SNR (dB)"] == spectrum().snr

        cells = next(csv.reader([payloads["csv"]], delimiter=";"))
        assert cells[0] == "0"
        assert len(cells[5].split(",")) == 256
        assert len(cells[6].split(",")) == 256

        # 512 half precision values of 3 bytes, plus the header fields
        assert len(payloads["cbor"]) < 2000
        row = cbor2.loads(payloads["cbor"], tag_hook=uncborify)
        assert row[:5] == ["0", T0, datetime.timedelta(seconds=300), 150_005_000.0, 1171.875]
        assert row[5] == spectrum().noise
        assert row[6] == spectrum().snr

    def test_unknown_type_is_skipped(self, client, consumer):
        consumer.add(object())  # type: ignore[arg-type]
        client.publish.assert_not_called()

    def test_default_client_id_is_the_hostname(self, consumer):
        import platform

        assert consumer.client_id == f"{platform.node()}-radiotracking"

    def test_client_id_is_configurable(self):
        with mock.patch("paho.mqtt.client.Client") as cls:
            c = MQTTConsumer("localhost", 1883, 1, 60, prefix="x", mqtt_client_id="debug-run")
        assert cls.call_args.kwargs["client_id"] == "debug-run"
        c.shutdown()

    def test_unexpected_disconnect_points_to_the_client_id(self, consumer, caplog):
        failure = mock.Mock(is_failure=True)
        with caplog.at_level(logging.WARNING):
            consumer._on_disconnect(None, None, None, failure, None)
        assert "--mqtt-client-id" in caplog.text
        assert consumer.client_id in caplog.text

    def test_regular_disconnect_is_not_logged(self, consumer, caplog):
        with caplog.at_level(logging.WARNING):
            consumer._on_disconnect(None, None, None, mock.Mock(is_failure=False), None)
            consumer.shutdown()
            consumer._on_disconnect(None, None, None, mock.Mock(is_failure=True), None)
        assert caplog.text == ""

    def test_publishing_does_not_wait_for_the_broker(self, client, consumer, make_signal):
        consumer.add(make_signal())
        assert client.publish.call_count == 3
        client.publish.return_value.wait_for_publish.assert_not_called()

    def test_queue_is_bounded(self, client, consumer):
        from radiotracking.consume import MQTT_MAX_QUEUED_MESSAGES

        client.max_queued_messages_set.assert_called_once_with(MQTT_MAX_QUEUED_MESSAGES)

    def test_shutdown_waits_for_the_last_message_bounded(self, client, consumer, make_signal):
        consumer.add(make_signal())
        consumer.shutdown()
        client.publish.return_value.wait_for_publish.assert_called_once_with(timeout=2.0)
        client.loop_stop.assert_called_once()

    def test_shutdown_without_messages_does_not_wait(self, client, consumer):
        consumer.shutdown()
        client.publish.return_value.wait_for_publish.assert_not_called()

    def test_publish_errors_do_not_propagate(self, client, consumer, make_signal):
        client.publish.side_effect = OSError("broker gone")
        consumer.add(make_signal())

    def test_no_publish_after_shutdown(self, client, consumer, make_signal):
        consumer.shutdown()
        consumer.shutdown()
        client.loop_stop.assert_called_once()
        consumer.add(make_signal())
        client.publish.assert_not_called()

    def test_log_records_are_not_published(self, client, consumer):
        logging.getLogger("radiotracking").warning("not for mqtt")
        client.publish.assert_not_called()


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
        assert len(names) == 5
        assert any(n.endswith("-matched.csv") for n in names)
        assert any(n.endswith("-state.csv") for n in names)
        assert any(n.endswith("-spectrum.csv") for n in names)
        assert any(n.endswith("-gain.csv") for n in names)
        assert not any("calibrated" in n for n in names)

        conn.q.put(make_signal())
        conn.step(datetime.timedelta(seconds=2))
        signal_csv = next(
            p for p in outdir.iterdir() if not p.name.endswith(("-matched.csv", "-state.csv", "-spectrum.csv", "-gain.csv"))
        )
        assert len(signal_csv.read_text().splitlines()) == 2

    def test_calibration_suffix(self, tmp_path):
        self.make(tmp_path, csv=True, calibrate=True)
        names = [p.name for p in (tmp_path / socket.gethostname() / "radiotracking").iterdir()]
        assert all(n.endswith("_calibration.csv") for n in names)

    def test_mqtt_is_skipped_in_calibration(self, tmp_path):
        with mock.patch("paho.mqtt.client.Client"):
            conn = self.make(tmp_path, mqtt=True, calibrate=True, mqtt_host="h", mqtt_port=1, mqtt_qos=1, mqtt_keepalive=1)
        assert conn.consumers == []

    def test_mqtt_consumer_is_no_log_handler_and_shutdown_stops_it(self, tmp_path):
        with mock.patch("paho.mqtt.client.Client") as cls:
            conn = self.make(tmp_path, mqtt=True, mqtt_host="h", mqtt_port=1, mqtt_qos=1, mqtt_keepalive=1)
        consumer = conn.consumers[0]
        assert isinstance(consumer, MQTTConsumer)
        assert consumer not in logging.getLogger("radiotracking").handlers
        conn.shutdown()
        conn.shutdown()
        cls.return_value.loop_stop.assert_called_once()

    def test_step_returns_on_timeout(self, tmp_path):
        assert self.make(tmp_path).step(datetime.timedelta(seconds=0.05)) is False

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

    def test_failing_consumer_does_not_starve_others(self, tmp_path, make_signal, caplog):
        conn = self.make(tmp_path)
        seen = []
        conn.consumers.extend(
            [
                mock.Mock(add=seen.append),
                mock.Mock(add=mock.Mock(side_effect=RuntimeError("boom"))),
                mock.Mock(add=seen.append),
            ]
        )
        conn.q.put(make_signal())
        with caplog.at_level(logging.ERROR):
            assert conn.step(datetime.timedelta(seconds=2)) is True
        assert len(seen) == 2
        assert "failed to consume" in caplog.text and "boom" in caplog.text

    def test_failing_consumer_does_not_break_following_steps(self, tmp_path, make_signal):
        conn = self.make(tmp_path)
        seen = []
        conn.consumers.extend([mock.Mock(add=mock.Mock(side_effect=RuntimeError("boom"))), mock.Mock(add=seen.append)])
        for _ in range(2):
            conn.q.put(make_signal())
            conn.step(datetime.timedelta(seconds=2))
        assert len(seen) == 2


class TestCalibratedFiles:
    def make(self, tmp_path, fullscale_dbm):
        return ProcessConnector(
            station="st",
            device=["0", "1"],
            calibrate=False,
            sig_stdout=False,
            match_stdout=False,
            path=str(tmp_path),
            csv=True,
            mqtt=False,
            fullscale_dbm=fullscale_dbm,
        )

    def test_calibrated_device_adds_columns_and_a_file(self, tmp_path):
        conn = self.make(tmp_path, [-20.0, None])
        outdir = tmp_path / socket.gethostname() / "radiotracking"
        names = sorted(p.name for p in outdir.iterdir())
        assert len(names) == 6
        assert any(n.endswith("-calibrated.csv") for n in names)

        signal_csv = next(p for p in outdir.iterdir() if p.name.count("-") == 2 and p.name.endswith(".csv"))
        header = signal_csv.read_text().splitlines()[0].split(";")
        assert header[-3:] == ["max (dBm)", "avg (dBm)", "noise (dBm)"]

        conn.q.put(CalibratedMatchedSignal(["0", "1"], T0, 150_100_000.0, 0.02, -100.0, None))
        conn.step(datetime.timedelta(seconds=2))
        matched = next(p for p in outdir.iterdir() if p.name.endswith("-calibrated.csv"))
        assert matched.read_text().splitlines()[0] == "Time;Frequency;Duration;0;1"
        assert matched.read_text().splitlines()[1].endswith(";-100.0;")

    def test_gain_message_is_written(self, tmp_path):
        conn = self.make(tmp_path, [])
        outdir = tmp_path / socket.gethostname() / "radiotracking"
        conn.q.put(GainMessage("0", T0, 64.5, 15, 14, 7, 0.0, 32.03, None, None))
        conn.step(datetime.timedelta(seconds=2))
        gain = next(p for p in outdir.iterdir() if p.name.endswith("-gain.csv"))
        lines = gain.read_text().splitlines()
        assert lines[0].split(";")[:3] == ["Device", "Time", "Gain (dB)"]
        assert lines[1].split(";")[2] == "64.5"
