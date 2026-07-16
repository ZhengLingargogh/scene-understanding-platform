"""Tests for CrossLoc calibration/pose file parsing."""

import numpy as np

from app.utils.crossloc_io import (
    parse_calibration_text,
    parse_pose_text,
    scale_intrinsics_for_resize,
)

NATURE_CAL = "480.0\n"
NATURE_POSE = """4.046813641045819421e-01 -9.100674533939241417e-01 8.949985374039573505e-02 -2.251370689459145069e+02
-8.703318696153674594e-01 -3.532693759164306502e-01 3.431081240242418451e-01 6.948017944982275367e+02
-2.806339791940270501e-01 -2.167440387016748571e-01 -9.350222411306660097e-01 2.058029693430289626e+02
0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00"""


def test_parse_nature_calibration():
    cal = parse_calibration_text(NATURE_CAL)
    assert cal["focal_length"] == 480.0
    assert cal["pp_x"] is None


def test_parse_nature_pose():
    pose = parse_pose_text(NATURE_POSE)
    assert pose.shape == (4, 4)
    assert pose[3, 3] == 1.0


def test_scale_intrinsics_720x480_to_height_480():
    cal = {"focal_length": 480.0, "pp_x": None, "pp_y": None}
    scaled = scale_intrinsics_for_resize(cal, orig_height=480.0, resized_width=720.0, resized_height=480.0)
    assert scaled["focal_length"] == 480.0
    assert scaled["pp_x"] == 360.0
    assert scaled["scale_factor"] == 1.0
