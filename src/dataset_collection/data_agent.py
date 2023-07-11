import gym
import mani_skill2.envs  # import to register all environments in gym
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer

from tqdm import tqdm
import numpy as np
import os
import gzip
import pickle
import math

env = gym.make("PickCube-v0", obs_mode="rgbd", control_mode="pd_ee_delta_pose")


class DataAgent:

    def __init__(self, env, env_id, control_mode, obs_mode="rgbd", render_mode="cameras", num_steps=100):
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.env_id = env_id
        self.env = env
        self.render_mode = render_mode
        self.obs = self.env.reset()
        self.done = False
        self.info = None
        self._cache = {
            "image_rgb": [],
            "extrinsic": [],
            "intrinsic": [],
            "depth": [],
            "name": [],
            "pose": [],
            "euler_angles": [],
        }

        self.num_steps = num_steps
        self.action = None
        self.episode_count = 0

    @staticmethod
    def _euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    def _init_action_dict(self):
        # Embodiment
        self._has_base = "base" in self.env.agent.controller.configs
        self._num_arms = sum("arm" in x for x in self.env.agent.controller.configs)
        self._has_gripper = any("gripper" in x for x in self.env.agent.controller.configs)
        self._gripper_action = 1
        self._EE_ACTION = 0.2
        if self._has_base:
            assert self.control_mode in ["base_pd_joint_vel_arm_pd_ee_delta_pose"]
            self._base_action = np.zeros([4])  # hardcoded
        else:
            self._base_action = np.zeros([0])

        # Parse end-effector action
        if (
                "pd_ee_delta_pose" in self.control_mode
                or "pd_ee_target_delta_pose" in self.control_mode
        ):
            self._ee_action = np.zeros([6])
        elif (
                "pd_ee_delta_pos" in self.control_mode
                or "pd_ee_target_delta_pos" in self.control_mode
        ):
            self._ee_action = np.zeros([3])
        else:
            raise NotImplementedError(self.control_mode)

    def change_env(self, new_env):
        self.env = new_env

    def _save_cache(self):
        self._cache['image_rgb'].append(self.obs['image']['hand_camera']['rgb'])
        self._cache['intrinsic'].append(self.obs['camera_param']['hand_camera']['intrinsic_cv'])
        self._cache['extrinsic'].append(self.obs['camera_param']['hand_camera']['extrinsic_cv'])
        self._cache['depth'].append(self.obs['image']['hand_camera']['depth'])
        self._cache['pose'].append(self.obs['extra']['tcp_pose'][0:3])
        roll, pitch, yaw = self._euler_from_quaternion(self.obs['extra']['tcp_pose'][3],
                                                       self.obs['extra']['tcp_pose'][4],
                                                       self.obs['extra']['tcp_pose'][5],
                                                       self.obs['extra']['tcp_pose'][6])
        self._cache['euler_angles'].append([roll, pitch, yaw])

    def store_cache(self, datasets_dir, data_name):
        if not os.path.exists(datasets_dir):
            os.mkdir(datasets_dir)
        data_name = f"{data_name}_episode_{self.episode_count}.pkl.gzip"
        data_file = os.path.join(datasets_dir, data_name)
        f = gzip.open(data_file, 'wb')
        pickle.dump(self._cache, f)
        f.close()
        print(f"data saved in location {datasets_dir} file: {data_name}")

    def generate_EE_action(self, direction: str):  # +y action
        self._init_action_dict()
        # End-effector
        if self._num_arms > 0:
            # Position
            if direction == "i":  # +x
                self._ee_action[0] = self._EE_ACTION
            elif direction == "k":  # -x
                self._ee_action[0] = -self._EE_ACTION
            elif direction == "j":  # +y
                self._ee_action[1] = self._EE_ACTION
            elif direction == "l":  # -y
                self._ee_action[1] = -self._EE_ACTION
            elif direction == "u":  # +z
                self._ee_action[2] = self._EE_ACTION
            elif direction == "o":  # -z
                self._ee_action[2] = -self._EE_ACTION

            # Rotation (axis-angle)
            if direction == "1":
                self._ee_action[3:6] = (1, 0, 0)
            elif direction == "2":
                self._ee_action[3:6] = (-1, 0, 0)
            elif direction == "3":
                self._ee_action[3:6] = (0, 1, 0)
            elif direction == "4":
                self._ee_action[3:6] = (0, -1, 0)
            elif direction == "5":
                self._ee_action[3:6] = (0, 0, 1)
            elif direction == "6":
                self._ee_action[3:6] = (0, 0, -1)
        action_dict = dict(base=self._base_action, arm=self._ee_action, gripper=self._gripper_action)
        action = env.agent.controller.from_action_dict(action_dict)
        return action

    def run_episode(self, dataset_dir, dataset_name, transformation_type="translation_X"):
        self.obs = self.env.reset(reconfigure=True)  # reset the environment
        self._init_action_dict()
        opencv_viewer = OpenCVViewer(exit_on_esc=False)
        for j in range(10):
            action = self.generate_EE_action("nothing")
            self.obs, _, self.done, self.info = self.env.step(action)
            self.render_camera(opencv_viewer)
        for i in tqdm(range(self.num_steps)):
            if transformation_type == "translation_X":
                if i > self.num_steps / 3:
                    self._EE_ACTION = 1.5 * self._EE_ACTION
                    action = self.generate_EE_action("k")
                else:
                    action = self.generate_EE_action("i")
            elif transformation_type == "translation_Y":
                if i > self.num_steps / 3:
                    self._EE_ACTION = 2 * self._EE_ACTION
                    action = self.generate_EE_action("l")
                else:
                    action = self.generate_EE_action("j")
            elif transformation_type == "translation_Z":
                if i > self.num_steps / 3:
                    self._EE_ACTION = 2 * self._EE_ACTION
                    action = self.generate_EE_action("o")
                else:
                    action = self.generate_EE_action("u")
            elif transformation_type == "rotation_X":
                if i > self.num_steps / 3:
                    self._EE_ACTION = 1.5 * self._EE_ACTION
                    action = self.generate_EE_action("2")
                else:
                    action = self.generate_EE_action("1")
            elif transformation_type == "rotation_Y":
                if i > self.num_steps / 3:
                    self._EE_ACTION = 1.5 * self._EE_ACTION
                    action = self.generate_EE_action("4")
                else:
                    action = self.generate_EE_action("3")
            elif transformation_type == "rotation_Z":
                if i > self.num_steps / 3:
                    self._EE_ACTION = 1.5 * self._EE_ACTION
                    action = self.generate_EE_action("6")
                else:
                    action = self.generate_EE_action("5")
            else:
                raise NotImplementedError(transformation_type)
            self.obs, _, self.done, self.info = self.env.step(action)
            self.render_camera(opencv_viewer)
            if i % 2 == 0:
                self._save_cache()
        self.episode_count += 1
        self.store_cache(dataset_dir, dataset_name)

    def render_camera(self, opencv_viewer):
        rendered_frame = self.env.render(mode='cameras')
        opencv_viewer.imshow(rendered_frame)

# env.seed(0)  # specify a seed for randomness
#
#
#
# cl
# done = False
# record_dir = "traj_data"
# env = RecordEpisode(env, record_dir, render_mode="cameras")
# obs = env.reset()
# for i in tqdm(range(100)):
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     env.render()
# obs = env.reset()
# for i in tqdm(range(100)):
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     env.render()
#
#
# env.close()
