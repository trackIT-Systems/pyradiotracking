import configparser
import io
import sys
from pathlib import Path

import pytest

from radiotracking.__main__ import Runner
from radiotracking.config import ArgConfParser

parser = Runner.parser
ETC_INI = Path(__file__).parent.parent / "etc" / "radiotracking.ini"


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


@pytest.mark.xfail(
    sys.version_info >= (3, 14),
    strict=True,
    reason="help texts are written as keys, configparser of Python 3.14 rejects the ':' in the schedule help; breaks etc/generate-example-config.py",
)
def test_write_config_with_help(tmp_path):
    args = parser.parse_args(["--config", str(tmp_path / "none.ini")])
    buf = io.StringIO()
    parser.write_config(args, buf, help=True)
    # configparser lowercases the comment "keys" on some Python versions
    assert "# center frequency to tune to (hz)" in buf.getvalue().lower()


@pytest.mark.xfail(
    sys.version_info >= (3, 13),
    strict=True,
    reason="immutable_args() calls the private argparse API _parse_known_args(), whose signature changed in Python 3.13; the only user was the removed config dashboard",
)
def test_immutable_args_are_those_given_on_the_command_line():
    assert set(parser.immutable_args(["-f", "150000000", "--csv"])) == {"center_freq", "csv"}


def test_station_default_is_hostname(tmp_path):
    import platform

    assert parser.parse_args(["--config", str(tmp_path / "none.ini")]).station == platform.node()


@pytest.mark.parametrize("section", ["optional arguments", "options"])
def test_optional_arguments_section_is_read_under_either_name(tmp_path, section):
    cfg = write(tmp_path, f"[{section}]\nstation = 'legacy'\nverbose = 2\nschedule = ['10:00-11:00']\n")
    args = parser.parse_args(["--config", cfg])
    assert args.station == "legacy"
    assert args.verbose == 2
    assert args.schedule == ["10:00-11:00"]


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
