import configparser
import io
import logging
from pathlib import Path

import pytest

from radiotracking.__main__ import Runner
from radiotracking.config import ArgConfParser

parser = Runner.parser
ETC_INI = Path(__file__).parent.parent / "etc" / "radiotracking.ini"
# the config of tsOS-vhf, if this is checked out as its submodule
TSOS_VHF_INI = next(
    (p / "boot/firmware/radiotracking.ini" for p in Path(__file__).parents if (p / "boot/firmware/radiotracking.ini").exists()),
    None,
)


def write(tmp_path, text):
    path = tmp_path / "radiotracking.ini"
    path.write_text(text)
    return str(path)


def known_dests():
    return {a.dest for g in parser._action_groups if isinstance(g.title, str) for a in g._group_actions}


def test_defaults_without_config_file(tmp_path):
    args = parser.parse_args(["--config", str(tmp_path / "missing.ini")])
    assert args.device == ["0"]
    assert args.center_freq == 150_150_000
    assert args.mqtt is False


def test_config_file_overrides_defaults(tmp_path):
    cfg = write(tmp_path, "[rtl-sdr]\ncenter_freq = 150000000\ndevice = ['0', '1']\ncalibration = [1.5, 2.5]\n")
    args = parser.parse_args(["--config", cfg])
    assert args.center_freq == 150_000_000
    assert args.device == ["0", "1"]
    assert args.calibration == [1.5, 2.5]


def test_command_line_overrides_config_file(tmp_path):
    cfg = write(tmp_path, "[rtl-sdr]\ncenter_freq = 150000000\n")
    args = parser.parse_args(["--config", cfg, "-f", "151000000"])
    assert args.center_freq == 151_000_000


def test_config_sections_of_removed_options_are_ignored(tmp_path):
    cfg = write(tmp_path, "[dashboard]\ndashboard = True\ndashboard_port = 8050\n[rtl-sdr]\ngain = 10.0\n")
    args = parser.parse_args(["--config", cfg])
    assert args.gain == 10.0
    assert "dashboard" not in vars(args)


def test_every_key_of_the_example_config_is_a_known_option():
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(ETC_INI)
    keys = {k for section in config.sections() for k in config[section] if not k.startswith("#")}
    assert keys, "example config is empty"
    assert keys <= known_dests()


def test_example_config_is_parseable():
    args = parser.parse_args(["--config", str(ETC_INI)])
    assert args.device


def test_write_and_read_roundtrip(tmp_path):
    args = parser.parse_args(["--config", str(tmp_path / "none.ini"), "-d", "0", "1", "-c", "1.0", "2.0", "--csv"])
    buf = io.StringIO()
    parser.write_config(args, buf)
    path = tmp_path / "out.ini"
    path.write_text(buf.getvalue())

    read = parser.read_config(str(path))
    for dest, value in read.items():
        assert vars(args)[dest] == value
    assert read["device"] == ["0", "1"]
    assert read["csv"] is True
    assert read["fft_window"] == "hamming"


def test_write_config_with_help(tmp_path):
    args = parser.parse_args(["--config", str(tmp_path / "none.ini")])
    buf = io.StringIO()
    parser.write_config(args, buf, help=True)
    assert "# center frequency to tune to (Hz)" in buf.getvalue()
    assert "# interval for publishing the noise and SNR per frequency (s); <=0 disables" in buf.getvalue()


def test_config_written_with_help_can_be_read_back(tmp_path):
    args = parser.parse_args(["--config", str(tmp_path / "none.ini"), "-d", "0", "1", "-c", "1.0", "2.0"])
    path = tmp_path / "out.ini"
    with open(path, "w") as f:
        parser.write_config(args, f, help=True)
    read = parser.read_config(str(path))
    assert read["device"] == ["0", "1"]
    assert read["calibration"] == [1.0, 2.0]


def test_percent_in_values_survives_roundtrip(tmp_path):
    args = parser.parse_args(["--config", str(tmp_path / "none.ini"), "--station", "100%"])
    path = tmp_path / "out.ini"
    with open(path, "w") as f:
        parser.write_config(args, f)
    assert parser.read_config(str(path))["station"] == "100%"


def test_analysis_defaults(tmp_path):
    args = parser.parse_args(["--config", str(tmp_path / "none.ini")])
    assert args.analysis_block_samples is None
    assert args.signal_min_duration_ms == 8
    assert args.signal_max_duration_ms == 50


def test_station_default_is_hostname(tmp_path):
    import platform

    assert parser.parse_args(["--config", str(tmp_path / "none.ini")]).station == platform.node()


@pytest.mark.parametrize("section", ["optional arguments", "options"])
def test_optional_arguments_section_is_read_under_either_name(tmp_path, section):
    cfg = write(tmp_path, f"[{section}]\nstation = 'legacy'\nverbose = 2\n")
    args = parser.parse_args(["--config", cfg])
    assert args.station == "legacy"
    assert args.verbose == 2


def test_current_section_name_wins_over_legacy_name(tmp_path):
    cfg = write(tmp_path, "[optional arguments]\nstation = 'old'\nverbose = 1\n[options]\nstation = 'new'\n")
    args = parser.parse_args(["--config", cfg])
    assert args.station == "new"
    assert args.verbose == 1


def test_legacy_config_of_tsos_vhf_is_read(tmp_path):
    cfg = write(
        tmp_path,
        "[optional arguments]\nverbose = 0\ncalibrate = False\nconfig = '/boot/firmware/radiotracking.ini'\n"
        "schedule = []\n[rtl-sdr]\ndevice = ['0']\ncalibration = [0.0]\n",
    )
    args = parser.parse_args(["--config", cfg])
    assert args.device == ["0"] and args.calibration == [0.0]


def test_arg_conf_parser_without_config_dest():
    p = ArgConfParser()
    p.add_argument("--x", default=1, type=int)
    assert p.parse_args(["--x", "2"]).x == 2


class TestRemovedOptions:
    def test_empty_schedule_of_old_configs_is_ignored_silently(self, tmp_path, caplog):
        cfg = write(tmp_path, "[optional arguments]\nschedule = []\n")
        with caplog.at_level(logging.WARNING):
            args = parser.parse_args(["--config", cfg])
        assert "schedule" not in vars(args)
        assert caplog.text == ""

    @pytest.mark.parametrize("section", ["options", "optional arguments"])
    def test_configured_schedule_is_ignored_with_a_warning(self, tmp_path, caplog, section):
        cfg = write(tmp_path, f"[{section}]\nschedule = ['10:00-11:00']\nstation = 'x'\n")
        with caplog.at_level(logging.WARNING):
            args = parser.parse_args(["--config", cfg])
        assert args.station == "x"
        assert "schedule" not in vars(args)
        assert "Ignoring 'schedule'" in caplog.text

    @pytest.mark.parametrize("value,warns", [("0", False), ("1", True)])
    def test_mqtt_verbose_of_old_configs(self, tmp_path, caplog, value, warns):
        cfg = write(tmp_path, f"[publish]\nmqtt_verbose = {value}\n")
        with caplog.at_level(logging.WARNING):
            args = parser.parse_args(["--config", cfg])
        assert "mqtt_verbose" not in vars(args)
        assert ("Ignoring 'mqtt_verbose'" in caplog.text) is warns
        assert "unknown" not in caplog.text

    @pytest.mark.parametrize("value,warns", [("0", False), ("60.0", True)])
    def test_status_interval_of_old_configs(self, tmp_path, caplog, value, warns):
        cfg = write(tmp_path, f"[rtl-sdr]\nstatus_interval = {value}\n")
        with caplog.at_level(logging.WARNING):
            args = parser.parse_args(["--config", cfg])
        assert "status_interval" not in vars(args)
        assert args.spectrum_interval_s == 300.0
        assert ("renamed to spectrum_interval_s" in caplog.text) is warns

    def test_schedule_is_not_a_command_line_option(self, tmp_path):
        with pytest.raises(SystemExit):
            parser.parse_args(["--config", str(tmp_path / "none.ini"), "--schedule", "10:00-11:00"])


def test_analysis_block_samples_is_read_from_the_config(tmp_path):
    cfg = write(tmp_path, "[analysis]\nanalysis_block_samples = 30208\n")
    assert parser.parse_args(["--config", cfg]).analysis_block_samples == 30208


class TestValueTypes:
    """Values of the config file are python literals, the option's type is applied like argparse does."""

    @pytest.mark.parametrize(
        "line,dest,expected",
        [
            ("[rtl-sdr]\nsdr_timeout_s = 2.0", "sdr_timeout_s", 2),
            ("[rtl-sdr]\ncenter_freq = 150.155e6", "center_freq", 150_155_000),
            ("[rtl-sdr]\nsample_rate = '300000'", "sample_rate", 300_000),
            ("[rtl-sdr]\ndevice = [0, 1]", "device", ["0", "1"]),
            ("[rtl-sdr]\ndevice = ('0',)", "device", ["0"]),
            ("[rtl-sdr]\ncalibration = [1, 2]", "calibration", [1.0, 2.0]),
            ("[rtl-sdr]\ngain = 40", "gain", 40.0),
            ("[analysis]\nfft_nperseg = 256.0", "fft_nperseg", 256),
            ("[analysis]\nanalysis_block_samples = None", "analysis_block_samples", None),
            ("[analysis]\nfft_window = ('kaiser', 8.0)", "fft_window", ("kaiser", 8.0)),
            ("[options]\nstation = 42", "station", "42"),
            # unquoted strings, as in the config files of the stations
            ("[options]\nstation = trackIT", "station", "trackIT"),
        ],
    )
    def test_value_is_converted(self, tmp_path, line, dest, expected):
        value = getattr(parser.parse_args(["--config", write(tmp_path, line + "\n")]), dest)
        assert value == expected
        assert type(value) is type(expected)

    @pytest.mark.parametrize(
        "line,message",
        [
            ("[rtl-sdr]\nsdr_timeout_s = 2.5", "'sdr_timeout_s' in [rtl-sdr]"),
            ("[rtl-sdr]\nsample_rate = 'fast'", "must be of type int"),
            ("[rtl-sdr]\ncalibration = ['x']", "must be of type float"),
            ("[rtl-sdr]\nsample_rate = fast", "strings need quotes"),
            ("[rtl-sdr]\ndevice = [a, b]", "strings need quotes"),
            ("[analysis]\nsnr_threshold_db = 5 dB", "'snr_threshold_db' in [analysis] of"),
        ],
    )
    def test_invalid_value_exits_with_a_message(self, tmp_path, capsys, line, message):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--config", write(tmp_path, line + "\n")])
        assert exc.value.code == 2
        assert message in capsys.readouterr().err


class TestUnknownOptions:
    def test_typo_is_reported(self, tmp_path, caplog):
        cfg = write(tmp_path, "[analysis]\nsnr_treshold_db = 10.0\n")
        with caplog.at_level(logging.WARNING):
            args = parser.parse_args(["--config", cfg])
        assert "Ignoring unknown option 'snr_treshold_db' in [analysis]" in caplog.text
        assert args.snr_threshold_db == 5.0

    def test_option_in_another_section_is_known(self, tmp_path, caplog):
        # options are only read from their own section, but are not unknown
        cfg = write(tmp_path, "[analysis]\ngain = 10.0\n")
        with caplog.at_level(logging.WARNING):
            parser.parse_args(["--config", cfg])
        assert "unknown" not in caplog.text

    def test_disabled_dashboard_of_old_configs_is_ignored_silently(self, tmp_path, caplog):
        cfg = write(tmp_path, "[dashboard]\ndashboard = False\ndashboard_host = 'localhost'\ndashboard_port = 8050\n")
        with caplog.at_level(logging.WARNING):
            parser.parse_args(["--config", cfg])
        assert caplog.text == ""

    def test_enabled_dashboard_warns(self, tmp_path, caplog):
        cfg = write(tmp_path, "[dashboard]\ndashboard = True\n")
        with caplog.at_level(logging.WARNING):
            parser.parse_args(["--config", cfg])
        assert "Ignoring 'dashboard' in [dashboard]" in caplog.text

    def test_missing_config_file_warns(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            args = parser.parse_args(["--config", str(tmp_path / "missing.ini")])
        assert "could not be read" in caplog.text
        assert args.device == ["0"]

    @pytest.mark.parametrize("path", [ETC_INI, TSOS_VHF_INI], ids=["example", "tsos-vhf"])
    def test_shipped_configs_are_read_without_warnings(self, path, caplog):
        if path is None:
            pytest.skip("not checked out as submodule of tsOS-vhf")
        with caplog.at_level(logging.WARNING):
            parser.parse_args(["--config", str(path)])
        assert caplog.text == ""


def test_spectrum_interval_is_read_from_the_config(tmp_path):
    cfg = write(tmp_path, "[rtl-sdr]\nspectrum_interval_s = 600\n")
    assert parser.parse_args(["--config", cfg]).spectrum_interval_s == 600.0


def test_unquoted_config_path_of_the_stations_is_read(tmp_path):
    """Issue #9: exported config files list ``config = /boot/firmware/radiotracking.ini`` without quotes."""
    cfg = write(tmp_path, "[optional arguments]\nconfig = /boot/firmware/radiotracking.ini\nstation = trackIT\n")
    assert parser.read_config(cfg) == {"config": "/boot/firmware/radiotracking.ini", "station": "trackIT"}
