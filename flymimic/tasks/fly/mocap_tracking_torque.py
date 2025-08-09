import collections
import os

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from flymimic.tasks.fly import SUITE


class Physics(mujoco.Physics):
    def qpos(self):
        return self.data.qpos.copy()

    def qvel(self):
        return np.clip(self.data.qvel.copy() / 10, -10, 10)

    def xpos(self):
        return self.data.xpos.copy()

    def femur_loc(self):
        return self.named.data.xpos["LFFemur"].copy()

    def tibia_loc(self):
        return self.named.data.xpos["LFTibia"].copy()

    def tarsus_loc(self):
        return self.named.data.xpos["LFTarsus1"].copy()

    def claw_loc(self):
        return self.named.data.xpos["LFTarsus5"].copy()

    def joint_angle(self, joint_name):
        return self.named.data.qpos[joint_name].copy()


class MoCapTask(base.Task):
    def __init__(
        self,
        clip,
        min_episode_steps,
        pose_rew_weight,
        vel_rew_weight,
        init_noise_scale,
        rew_threshold,
        test,
        play,
        random,
        joint_ids=[0, 1, 2, 3, 4, 5, 6],
        body_ids=[21, 22, 23, 27],
    ):
        super().__init__(random=random)
        self._min_episode_steps = min_episode_steps
        self._pose_rew_weight = pose_rew_weight
        self._vel_rew_weight = vel_rew_weight
        self._init_noise_scale = init_noise_scale
        self._rew_threshold = rew_threshold
        self._test = test
        self._play = play
        self._mocap_index = 0
        self._max_mocap_index = 1
        self._joint_ids = joint_ids
        self._body_ids = body_ids
        self.initialize_clip(clip)

    def initialize_clip(self, clip):
        path = os.path.dirname(__file__) + "/../../assets/mocap/"

        # Joints.
        qpos_path = path + "qpos/" + clip + ".npy"
        self._mocap_qpos = np.load(qpos_path)

        # Velocities (approximated).
        qvel_path = path + "qvel/" + clip + ".npy"
        self._mocap_qvel = np.load(qvel_path)

        # Xipos.
        xpos_path = path + "xipos/" + clip + ".npy"
        self._mocap_xpos = np.load(xpos_path)

        self._clip_length = self._mocap_qpos.shape[0]

        self._num_joints = len(self._mocap_qpos[0])
        self._num_bodies = len(self._mocap_xpos[0])

    def initialize_episode(self, physics):
        if self._test:
            self._mocap_index = 0
        else:
            self._mocap_index = self.random.randint(
                self._clip_length - self._min_episode_steps
            )
        self._max_mocap_index = self._clip_length - 1

        # Joints.
        target_qpos = self._mocap_qpos[self._mocap_index]
        physics.data.qpos[self._joint_ids] = target_qpos
        if not self._test and self._init_noise_scale:
            physics.data.qpos[self._joint_ids] += self.random.normal(
                0, self._init_noise_scale, self._num_joints
            )

        # Velocities.
        target_qvel = self._mocap_qvel[self._mocap_index]
        physics.data.qvel[self._joint_ids] = target_qvel

    def after_step(self, physics):
        self._mocap_index += 1

        if self._play:
            target_qpos = self._mocap_qpos[self._mocap_index]
            physics.data.qpos[self._joint_ids] = target_qpos
            physics.data.qvel[self._joint_ids] = 0
            physics.forward()

        # Xipos.
        target_xpos = self._mocap_xpos[self._mocap_index]
        xpos = physics.data.xpos[self._body_ids]
        xpos_dists = np.mean(np.linalg.norm(target_xpos - xpos, axis=-1))
        xpos_rew = np.exp(-self._pose_rew_weight * xpos_dists)
        # Qpos
        target_qpos = self._mocap_qpos[self._mocap_index]
        qpos = physics.data.qpos[self._joint_ids]
        qpos_dists = np.mean(np.linalg.norm(target_qpos - qpos, axis=-1))
        qpos_rew = np.exp(-self._pose_rew_weight * qpos_dists)
        # Qvel
        target_qvel = self._mocap_qvel[self._mocap_index]
        qvel = physics.data.qvel[self._joint_ids]
        qvel_dists = np.mean(np.linalg.norm(target_qvel - qvel, axis=-1))
        qvel_rew = np.exp(-self._vel_rew_weight * qvel_dists)

        # print(
        #     "Xpos reward: ", xpos_rew,
        #     "Qpos reward: ", qpos_rew,
        #     "Total reward: ", xpos_rew * qpos_rew
        # )

        # Clip the reward
        # Mean of three rewards
        self._reward = float(np.clip((qpos_rew + xpos_rew + qvel_rew) / 3, 0.0, 1.0))
        # print("Reward: ", self._reward)
        # self._reward = float(np.clip(xpos_rew * qpos_rew, 0.0, 1.0))

    def get_observation(self, physics):
        obs = collections.OrderedDict()

        obs["femur_loc"] = physics.femur_loc()
        obs["tibia_loc"] = physics.tibia_loc()
        obs["tarsus_loc"] = physics.tarsus_loc()
        obs["claw_loc"] = physics.claw_loc()
        obs["qpos"] = physics.qpos()[self._joint_ids]
        obs["qvel"] = physics.qvel()[self._joint_ids]

        obs["time_left"] = np.array([1.0 - self._mocap_index / self._max_mocap_index])

        return obs

    def get_reward(self, physics):
        return self._reward

    def get_termination(self, physics):
        """
        Joint limits
        "joint_LFCoxa_roll": (-np.pi * 0.5, np.pi * 0.5),
        "joint_LFCoxa_yaw": (-np.pi * 0.5, np.pi * 0.5),
        "joint_LFCoxa_pitch": (-np.pi * 0.5, np.pi * 0.5),
        "joint_LFFemur_pitch": (-np.pi, 0),
        "joint_LFFemur_roll":(-np.pi * 0.1, np.pi * 0.1),
        "joint_LFTibia_pitch":(0, np.pi),
        """

        if self._reward < self._rew_threshold and not self._test:
            return True

        # if physics.joint_angle('joint_LFTrochanter_pitch') > 0 and not self._test:
        #     return True

        # if physics.joint_angle('joint_LFTibia_pitch') < 0 and not self._test:
        #     return True

        if self._mocap_index + 1 >= self._clip_length:
            return True


@SUITE.add("benchmarking")
def mocap_tracking_torque(
    clip="0002",
    min_episode_steps=20,
    pose_rew_weight=5,
    vel_rew_weight=3,
    environment_kwargs=None,
    rew_threshold=0.01,
    init_noise_scale=0.02,
    test=False,
    play=False,
    random=None,
    model_name="best_combined_cvt3_torque",
    control_timestep=0.002,  # 500 Hz
):
    task = MoCapTask(
        clip=clip,
        min_episode_steps=min_episode_steps,
        pose_rew_weight=pose_rew_weight,
        vel_rew_weight=vel_rew_weight,
        rew_threshold=rew_threshold,
        init_noise_scale=init_noise_scale,
        test=test,
        play=play,
        random=random,
    )

    path = os.path.dirname(__file__)
    path += f"/../../assets/models/{model_name}.xml"
    physics = Physics.from_xml_path(path)
    environment_kwargs = environment_kwargs or {}
    env = control.Environment(
        physics,
        task,
        time_limit=10000,
        control_timestep=control_timestep,
        **environment_kwargs,
    )

    return env
