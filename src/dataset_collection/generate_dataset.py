import argparse

import gym
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import math
import sapien.core as sapien
from mani_skill2 import make_box_space_readable
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2 import ASSET_DIR
from mani_skill2.utils.registration import register_env
from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
import gzip
import os
import pickle
from mani_skill2.utils.sapien_utils import look_at

MS1_ENV_IDS = [
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "PushChair-v1",
    "MoveBucket-v1",
]


def plot_img(img, title=None):
    plt.figure(figsize=(10,6))
    if title is not None:
        plt.title(title)
    plt.imshow(img)
    plt.show()





def euler_from_quaternion(x, y, z, w):
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

pose = look_at([1, -1, 0.5], [0, 0, 0])
env = gym.make(
    "PickCube-v0",
    camera_cfgs=dict(texture_names=("Color", "Position", "Segmentation")),
    render_camera_cfgs=dict(p=pose.p, q=pose.q),
)
plot_img(env.render("cameras"))
env.close()
del env
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, required=True)
    parser.add_argument("-o", "--obs-mode", type=str)
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", type=str, default="cameras")
    parser.add_argument("--enable-sapien-viewer", action="store_true")
    parser.add_argument("--record-dir", type=str)
    args, opts = parser.parse_known_args()

    # Parse env kwargs
    print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    return args

def show_camera_view(obs_camera, title):
    rgb, depth = obs_camera['rgb'], obs_camera['depth']
    plt.imshow(rgb) # for color
    #plt.imshow(depth[:,:, 0], cmap="gray") # for gray scale
    plt.show()

@register_env("PickYCBInReplicaCAD-v0", max_episode_steps=200, override=True)
class PickYCBInReplicaCAD(PickCubeEnv):
    def _load_actors(self):
        # Load YCB objects 
        # It is the same as in PickSingleYCB-v0, just for illustration here
        builder = self._scene.create_actor_builder()
        model_dir = ASSET_DIR / "mani_skill2_ycb/models/013_apple" # change object here
        scale = self.cube_half_size / 0.01887479572529618
        collision_file = str(model_dir / "collision.obj")
        builder.add_multiple_collisions_from_file(
            filename=collision_file, scale=scale, density=1000
        )
        visual_file = str(model_dir / "textured.obj")
        builder.add_visual_from_file(filename=visual_file, scale=scale)
        self.obj = builder.build(name="apple")

        # Add a goal indicator (visual only)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

        # -------------------------------------------------------------------------- #
        # Load static scene
        # -------------------------------------------------------------------------- #
        builder = self._scene.create_actor_builder()
        path = f"{ASSET_DIR}/hab2_bench_assets/stages/Baked_sc1_staging_00.glb"
        pose = sapien.Pose(q=[0.707, 0.707, 0, 0])  # y-axis up for Habitat scenes
        # NOTE: use nonconvex collision for static scene
        builder.add_nonconvex_collision_from_file(path, pose)
        builder.add_visual_from_file(path, pose)
        self.arena = builder.build_static()
        # Add offset to place the workspace at... (uncomment the background you want and comment the other offset)
        #offset = np.array([-1.9, 2, 0.9]) # another shelf (awkward angle, need to rotate camera)
        #offset = np.array([0.5, -0.2, 0.5]) # xyz z for height 0.5, 0, 0.5 or 0.7, 0, 0.5 couch
        #offset = np.array([2.3, 1.4, 0.5]) # stairs
        #offset = np.array([-1.5, -1, 0.3]) # carpet
        #offset = np.array([3.8, -0.8, 0.8]) # blue shelf
        #offset = np.array([2.5, -6.5, 0.9]) # shelf (need to rotate camera) 
        #offset = np.array([1.3, 3.7, 0.5]) # dark room
        #offset = np.array([4.2, 0.5, 0.8]) # bicycle
        #offset = np.array([4.1, -5.3, 0.9]) # corner of a sofa 
        offset = np.array([2.5, -5, 0.9]) # carpet
        self.arena.set_pose(sapien.Pose(-offset))

    def initialize_episode(self):
        super().initialize_episode()

        # Rotate the robot for better visualization
        self.agent.robot.set_pose(
            sapien.Pose([0, -0.56, 0], [0.707, 0, 0, 0.707]) #original

        )

    def _register_render_cameras(self):
        cam_cfg = super()._register_render_cameras()
        cam_cfg.p = cam_cfg.p + [0.5, 0.5, -0.095]
        cam_cfg.fov = 1.5
        return cam_cfg

def store_data(data, datasets_dir="./test_data"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data_rotation_apple.pkl.gzip')
    f = gzip.open(data_file, 'wb')
    pickle.dump(data, f)
    f.close()

def transform_points3d(obs, image_points = [443,221]):
    """
    Transform the points from image 1 to image 2.
    :param image1_data: The data of image 1.
    :param image2_data: The data of image 2.
    :param image1_points: The points in image 1.
    :return: The points in image 2.
    """

    # # Transform the points
    depth = float(obs['image']['hand_camera']['depth'][image_points[1]][image_points[0]])
    image1_points_h = np.array([image_points[0] * depth, image_points[1] * depth, depth])
    image1_points_camera = np.matmul(np.linalg.inv(obs['camera_param']['hand_camera']['intrinsic_cv']), image1_points_h)
    image1_points_camera_h = np.append(image1_points_camera, 1)
    image1_points_world_h = np.matmul(np.linalg.inv(obs['camera_param']['hand_camera']['extrinsic_cv']), image1_points_camera_h)

    image_points_3d = image1_points_world_h[:3]/image1_points_world_h[3]

    return image_points_3d

def main():
    make_box_space_readable()
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()

    if args.env_id in MS1_ENV_IDS:
        if args.control_mode is not None and not args.control_mode.startswith("base"):
            args.control_mode = "base_pd_joint_vel_arm_" + args.control_mode
 
    env: BaseEnv = gym.make( # default background (comment it out if you want to use custom background)
        args.env_id,
        obs_mode=args.obs_mode,
        #model_ids="002_master_chef_can", # Only for PickYCB
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_camera_cfgs=dict(width=640, height=480),
        #camera_cfgs={"add_segmentation": True}, 
        #shader_dir="rt", # only if PC supports ray tracing
        #render_config={"rt_samples_per_pixel": 2, "rt_use_denoiser": False}, # only if PC supports ray tracing
        #bg_name="minimal_bedroom", # optional background
        **args.env_kwargs
    )



    env = gym.make( # custom habitat2 background it overwrites the above env (comment it out if you want to use default background)
        "PickYCBInReplicaCAD-v0",
        obs_mode=args.obs_mode,

        render_camera_cfgs=dict(width=640, height=480),
        camera_cfgs=dict(hand_camera=dict(width=512, height=512)),
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        **args.env_kwargs)
    
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, render_mode=args.render_mode)

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)

    obs = env.reset()
    after_reset = True

    # Viewer
    if args.enable_sapien_viewer:
        env.render(mode="human")
    opencv_viewer = OpenCVViewer(exit_on_esc=False)

    def render_wait():
        if not args.enable_sapien_viewer:
            return
        while True:
            sapien_viewer = env.render(mode="human")
            if sapien_viewer.window.key_down("0"):
                break

    # Embodiment
    has_base = "base" in env.agent.controller.configs
    num_arms = sum("arm" in x for x in env.agent.controller.configs)
    has_gripper = any("gripper" in x for x in env.agent.controller.configs)
    gripper_action = 1
    EE_ACTION = 0.1
    counter = 0

    samples = {
        "image_rgb": [],
        "extrinsic": [],
        "intrinsic": [],
        "depth": [],
        "name": [],
        "pose": [],
        "euler_angles": [],
    }
    while True:
        # -------------------------------------------------------------------------- #
        # Visualization
        # -------------------------------------------------------------------------- #
        if args.enable_sapien_viewer:
            env.render(mode="human")

        render_frame = env.render(mode=args.render_mode)

        if after_reset:
            after_reset = False
            # Re-focus on opencv viewer
            if args.enable_sapien_viewer:
                opencv_viewer.close()
                opencv_viewer = OpenCVViewer(exit_on_esc=False)

        # -------------------------------------------------------------------------- #
        # Interaction
        # -------------------------------------------------------------------------- #
        # Input
        key = opencv_viewer.imshow(render_frame)
        if has_base:
            assert args.control_mode in ["base_pd_joint_vel_arm_pd_ee_delta_pose"]
            base_action = np.zeros([4])  # hardcoded
        else:
            base_action = np.zeros([0])

        # Parse end-effector action
        if (
            "pd_ee_delta_pose" in args.control_mode
            or "pd_ee_target_delta_pose" in args.control_mode
        ):
            ee_action = np.zeros([6])
        elif (
            "pd_ee_delta_pos" in args.control_mode
            or "pd_ee_target_delta_pos" in args.control_mode
        ):
            ee_action = np.zeros([3])
        else:
            raise NotImplementedError(args.control_mode)

        # Base
        if has_base:
            if key == "w":  # forward
                base_action[0] = 1
            elif key == "s":  # backward
                base_action[0] = -1
            elif key == "a":  # left
                base_action[1] = 1
            elif key == "d":  # right
                base_action[1] = -1
            elif key == "q":  # rotate counter
                base_action[2] = 1
            elif key == "e":  # rotate clockwise
                base_action[2] = -1
            elif key == "z":  # lift
                base_action[3] = 1
            elif key == "x":  # lower
                base_action[3] = -1

        # End-effector
        if num_arms > 0:
            # Position
            if key == "i":  # +x
                ee_action[0] = EE_ACTION
            elif key == "k":  # -x
                ee_action[0] = -EE_ACTION
            elif key == "j":  # +y
                ee_action[1] = EE_ACTION
            elif key == "l":  # -y
                ee_action[1] = -EE_ACTION
            elif key == "u":  # +z
                ee_action[2] = EE_ACTION
            elif key == "o":  # -z
                ee_action[2] = -EE_ACTION

            # Rotation (axis-angle)
            if key == "1":
                ee_action[3:6] = (1, 0, 0)
            elif key == "2":
                ee_action[3:6] = (-1, 0, 0)
            elif key == "3":
                ee_action[3:6] = (0, 1, 0)
            elif key == "4":
                ee_action[3:6] = (0, -1, 0)
            elif key == "5":
                ee_action[3:6] = (0, 0, 1)
            elif key == "6":
                ee_action[3:6] = (0, 0, -1)

        # Gripper
        if has_gripper:
            if key == "f":  # open gripper
                gripper_action = 1
            elif key == "g":  # close gripper
                gripper_action = -1

        # Other functions
        if key == "0":  # switch to SAPIEN viewer
            render_wait()
        elif key == "r":  # reset env
            obs = env.reset()
            gripper_action = 1
            after_reset = True
            continue
        elif key == None:  # exit
            break

        # Visualize observation
        if key == "v":
            if "rgbd" in env.obs_mode:
                from itertools import chain

                from mani_skill2.utils.visualization.misc import (
                    observations_to_images,
                    tile_images,
                )

                images = list(
                    chain(*[observations_to_images(x) for x in obs["image"].values()])
                )
                render_frame = tile_images(images)
                opencv_viewer.imshow(render_frame)
            elif "pointcloud" in env.obs_mode:
                import trimesh

                xyzw = obs["pointcloud"]["xyzw"]
                mask = xyzw[..., 3] > 0
                rgb = obs["pointcloud"]["rgb"]
                if "robot_seg" in obs["pointcloud"]:
                    robot_seg = obs["pointcloud"]["robot_seg"]
                    rgb = np.uint8(robot_seg * [11, 61, 127])
                trimesh.PointCloud(xyzw[mask, :3], rgb[mask]).show()

        # -------------------------------------------------------------------------- #
        # Post-process action
        # -------------------------------------------------------------------------- #
        if args.env_id in MS1_ENV_IDS:
            action_dict = dict(
                base=base_action,
                right_arm=ee_action,
                right_gripper=gripper_action,
                left_arm=np.zeros_like(ee_action),
                left_gripper=np.zeros_like(gripper_action),
            )
            action = env.agent.controller.from_action_dict(action_dict)
        else:
            action_dict = dict(base=base_action, arm=ee_action, gripper=gripper_action)
            action = env.agent.controller.from_action_dict(action_dict)

        obs, reward, done, info = env.step(action)
        if key == "g":
            go_to_pose(env, pose)
        if key == "8":
            counter +=1
            img_array = obs['image']['hand_camera']['rgb']
            samples['intrinsic'].append(obs['camera_param']['hand_camera']['intrinsic_cv'])
            samples['extrinsic'].append(obs['camera_param']['hand_camera']['extrinsic_cv'])
            samples['image_rgb'].append(img_array)
            samples['depth'].append(obs['image']['hand_camera']['depth'])
            samples['name'].append(f"test_{counter:02}.png")
            samples['pose'].append(obs['extra']['tcp_pose'][0:3])
            roll, pitch, yaw = euler_from_quaternion(obs['extra']['tcp_pose'][3], obs['extra']['tcp_pose'][4],
                                                     obs['extra']['tcp_pose'][5], obs['extra']['tcp_pose'][6])
            samples['euler_angles'].append([roll, pitch, yaw])
            print("image data recorded", samples['name'][-1])

        if key == "s":
            store_data(samples, "./test_data")
            print("data stored")
        roll, pitch, yaw = euler_from_quaternion(obs['extra']['tcp_pose'][3], obs['extra']['tcp_pose'][4],
                                                     obs['extra']['tcp_pose'][5], obs['extra']['tcp_pose'][6])
        print("roll, pitch, yaw", roll, pitch, yaw)
        print("tcp_pose", obs['extra']['tcp_pose'][0:3])



    print('obs', obs)
    print('info', info)

    show_camera_view(obs["image"]["hand_camera"], "Hand Camera View")

    env.close()

def go_to_pose(env, pose):
    action = np.zeros(6)
    action[0:3] = pose[0:3]
    action[3:6] = pose[3:6]
    obs, reward, done, info = env.step(action)
    return obs, reward, done, info

if __name__ == "__main__":
    main()
