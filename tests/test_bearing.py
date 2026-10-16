import math

import pytest

from radiotracking import MatchedSignal
from radiotracking.bearing import Gottwald2019Bearings

from .conftest import T0


def msig(*avgs):
    return MatchedSignal([str(i) for i in range(len(avgs))], T0, 150e6, 0.02, *avgs)


def test_missing_antenna_is_discarded():
    assert Gottwald2019Bearings().get_angle(msig(-40, None, -50, -60)) is None


def test_bearing_is_finite():
    angle = Gottwald2019Bearings().get_angle(msig(-40, -45, -60, -50))
    assert angle is not None and math.isfinite(angle)


@pytest.mark.xfail(
    strict=True,
    reason="the louder neighbour is chosen by comparing (index, avg) tuples, so the antenna index decides instead of the power",
)
def test_louder_neighbour_selects_the_sector():
    # antenna 1 is loudest, its louder neighbour is antenna 0 (-45 vs -50): bearing is between antennas 0 and 1
    angle = Gottwald2019Bearings().get_angle(msig(-45, -40, -50, -60))
    assert 0 <= angle < 90


@pytest.mark.xfail(strict=True, reason="acos() is fed a value outside [-1, 1] when the antennas differ by more than signal_maximum")
def test_large_level_difference_does_not_raise():
    Gottwald2019Bearings().get_angle(msig(-80, -40, -90, -90))
