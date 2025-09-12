from pathlib import Path
import pytest

from dm_control import mujoco

import flymimic

def test_mujoco_assets():
    mjcf_assets = Path(flymimic.__file__) / "assets" / "models"

    for asset in mjcf_assets.glob("*.xml"):
        assert asset.exists()
        mujoco.Physics.from_xml_path(str(asset))

def test_mocap_assets():
    mocap_assets = Path(flymimic.__file__) / "assets" / "mocap"

    for asset in mocap_assets.rglob("*.npy"):
        assert asset.exists()
        if "qpos" in str(asset) or "qvel" in str(asset):
            assert np.load(asset).shape[1] == 7  # Number of joints
        elif "xipos" in str(asset):
            assert np.load(asset).shape[1] == 5  # Number of bodies
