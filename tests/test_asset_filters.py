"""Isaac-free: object / container size gates."""

from pathlib import Path

import numpy as np

from sim.asset_filters import (
    container_can_hold,
    horizontal_width_m,
    merge_exclude_names,
    object_fits_gripper,
    stem_or_category_banned,
)


def test_gripper_width_gate():
    assert object_fits_gripper([0.05, 0.04, 0.06], 0.075)
    assert not object_fits_gripper([0.085, 0.08, 0.08], 0.075)  # tomato-sized
    assert horizontal_width_m([0.03, 0.09, 0.05]) == 0.09


def test_container_hold_gate():
    fruit = np.array([0.05, 0.05, 0.06])
    bowl = np.array([0.12, 0.12, 0.08])
    plate = np.array([0.20, 0.20, 0.02])
    assert container_can_hold(fruit, bowl)
    assert not container_can_hold(fruit, plate)  # too shallow
    tiny = np.array([0.04, 0.04, 0.08])
    assert not container_can_hold(fruit, tiny)  # too narrow
    # Open bowl: fruit may stick above the rim — XY only.
    shallow_bowl = np.array([0.15, 0.15, 0.042])
    tall_fruit = np.array([0.061, 0.060, 0.070])
    assert not container_can_hold(tall_fruit, shallow_bowl, min_clearance_m=0.01)
    assert container_can_hold(
        tall_fruit, shallow_bowl, min_clearance_m=0.01, open_top=True
    )
    assert not container_can_hold(
        tall_fruit, tiny, min_clearance_m=0.01, open_top=True
    )  # still too narrow


def test_container_needs_finger_room_across_y():
    """Closed fingers span the object + two pads along world Y."""
    potato = np.array([0.066, 0.066, 0.090])
    lime = np.array([0.061, 0.056, 0.065])
    narrow_box = np.array([0.242, 0.122, 0.120])
    wide_box = np.array([0.236, 0.167, 0.101])
    assert container_can_hold(potato, narrow_box)  # old rule: object alone fits
    assert not container_can_hold(potato, narrow_box, finger_clearance_m=0.06)
    assert container_can_hold(lime, narrow_box, finger_clearance_m=0.06)
    assert container_can_hold(potato, wide_box, finger_clearance_m=0.06)


def test_exclude_prefix_and_category():
    ban = merge_exclude_names(["tomato", "plate"], {"beer"})
    assert "tomato" in ban and "plate" in ban and "beer" in ban
    assert stem_or_category_banned(Path("/o/tomato/tomato03.usd"), ban)
    assert stem_or_category_banned(Path("/o/plate/plate02.usd"), ban)
    assert not stem_or_category_banned(Path("/o/apple/apple00.usd"), ban)
