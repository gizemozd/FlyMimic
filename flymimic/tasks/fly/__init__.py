# flake8: noqa
from dm_control.utils import containers

SUITE = containers.TaggedTasks()

from flymimic.tasks.fly import mocap_tracking_muscle, mocap_tracking_torque
