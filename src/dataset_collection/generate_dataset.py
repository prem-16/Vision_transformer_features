import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt
import math
import sapien.core as sapien
from mani_skill2 import make_box_space_readable
from mani_skill2.envs.pick_and_place.pick_clutter import PickClutterEnv
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2 import ASSET_DIR
from mani_skill2.utils.registration import register_env
from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.sapien_utils import set_render_material
from src.dataset_collection.helpers import store_data
from typing import List


from src.dataset_collection.data_agent import DataAgent
MS1_ENV_IDS = [
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "PushChair-v1",
    "MoveBucket-v1",
]





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

@register_env("DR-PickCube-v0", max_episode_steps=100, override=True)
class DomainRandomizationPickCubeEnv(PickCubeEnv):
    def _initialize_actors(self):
        super()._initialize_actors()

        # Example: randomize friction
        friction = self._episode_rng.uniform(0.5, 1.0)
        phys_mtl = self._scene.create_physical_material(
            static_friction=friction, dynamic_friction=friction, restitution=0.1
        )
        for cs in self.obj.get_collision_shapes():
            cs.set_physical_material(phys_mtl)

        # Example: randomize damping
        linear_damping = self._episode_rng.uniform(0, 1.0)
        angular_damping = self._episode_rng.uniform(0, 1.0)
        self.obj.set_damping(linear_damping, angular_damping)

        # Example: randomize color
        color = self._episode_rng.uniform(0.5, 1.0, size=3)
        for vb in self.obj.get_visual_bodies():
            for rs in vb.get_render_shapes():
                set_render_material(rs.material, color=np.hstack([color, 1.0]))

@register_env("PickMultiYCBInReplicaCAD-v0", max_episode_steps=200, override=True)
class PickMultiYCBInReplicaCAD(PickClutterEnv):
    DEFAULT_EPISODE_JSON = "src/dataset_collection/data/{ASSET_DIR}/pick_clutter/ycb_train_5k.json.gz"
    DEFAULT_EPISODE_JSON = "{ASSET_DIR}/ycb_train_5k.json.gz"
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_ycb"
    DEFAULT_MODEL_JSON = "info_pick_v0.json"

    def _load_actors(self):
        self.bbox_sizes = []
        self.objs: List[sapien.Actor] = []
        object_name = ["025_mug", "017_orange","024_bowl" ,"011_banana" , "004_sugar_box", "002_master_chef_can","077_rubiks_cube", "035_power_drill", "005_tomato_soup_can", "003_cracker_box"]
        # Add Multiple objects
        for object in object_name:
            builder = self._scene.create_actor_builder()
            model_dir = ASSET_DIR / "mani_skill2_ycb/models" / object
            collision_file = str(model_dir / "collision.obj")
            builder.add_multiple_collisions_from_file(
                filename=collision_file, density=1000
            )
            visual_file = str(model_dir / "textured.obj")
            builder.add_visual_from_file(filename=visual_file)

            obj = builder.build(name=object)
            self.objs.append(obj)

            bbox = self.model_db[object]["bbox"]
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.bbox_sizes.append(bbox_size)
        self.target_site = self._build_sphere_site(
            0.01, color=(1, 1, 0), name="_target_site"
        )
        self.goal_site = self._build_sphere_site(
            0.01, color=(0, 1, 0), name="_goal_site"
        )

        # -------------------------------------------------------------------------- #
        # Load static scene
        # -------------------------------------------------------------------------- #
        builder = self._scene.create_actor_builder()
        path = f"src/dataset_collection/data/{ASSET_DIR}/hab2_bench_assets/stages/Baked_sc1_staging_00.glb"
        pose = sapien.Pose(q=[0.707, 0.707, 0, 0])  # y-axis up for Habitat scenes
        # NOTE: use nonconvex collision for static scene
        builder.add_nonconvex_collision_from_file(path, pose)
        builder.add_visual_from_file(path, pose)
        self.arena = builder.build_static()
        offsets = [np.array([2.5, -5, 0.9]), np.array([0.5, -0.2, 0.5]), np.array([2.3, 1.4, 0.5])]
        # Add offset to place the workspace at... (uncomment the background you want and comment the other offset)
        # offset = np.array([-1.9, 2, 0.9]) # another shelf (awkward angle, need to rotate camera)
        # offset = np.array([0.5, -0.2, 0.5]) # xyz z for height 0.5, 0, 0.5 or 0.7, 0, 0.5 couch
        # offset = np.array([2.3, 1.4, 0.5]) # stairs
        # offset = np.array([-1.5, -1, 0.3]) # carpet
        offset = np.array([-1.5, -1, 0.3])  # carpet
        # offset = np.array([2.5, -6.5, 0.9]) # shelf (need to rotate camera)
        # offset = np.array([1.3, 3.7, 0.5]) # dark room
        # offset = np.array([4.2, 0.5, 0.8]) # bicycle
        # offset = np.array([4.1, -5.3, 0.9]) # corner of a sofa
        # offset = np.array([2.5, -5, 0.9]) # carpet
        self.arena.set_pose(sapien.Pose(-offset))

    def initialize_episode(self):
        super().initialize_episode()

        # Rotate the robot for better visualization
        self.agent.robot.set_pose(
            sapien.Pose([0, -0.56, 0], [0.707, 0, 0, 0.707])  # original

        )

    def _register_render_cameras(self):
        cam_cfg = super()._register_render_cameras()
        cam_cfg.p = cam_cfg.p + [0.5, 0.5, -0.095]
        cam_cfg.fov = 1.5
        return cam_cfg

@register_env("PickYCBInReplicaCAD-v0", max_episode_steps=200, override=True)
class PickYCBInReplicaCAD(PickCubeEnv):
    def _load_actors(self):
        # Load YCB objects 
        # It is the same as in PickSingleYCB-v0, just for illustration here
        builder = self._scene.create_actor_builder()
        object_name =np.random.choice(["025_mug", "017_orange","024_bowl" ,"011_banana" , "004_sugar_box", "037_scissors" , "072-b_toy_airplane", "077_rubiks_cube"])
        model_dir = ASSET_DIR / "mani_skill2_ycb/models" / object_name# change object here
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
        offsets = [np.array([2.5, -5, 0.9]) ,np.array([0.5, -0.2, 0.5]),np.array([2.3, 1.4, 0.5])]
        # Add offset to place the workspace at... (uncomment the background you want and comment the other offset)
        #offset = np.array([-1.9, 2, 0.9]) # another shelf (awkward angle, need to rotate camera)
        #offset = np.array([0.5, -0.2, 0.5]) # xyz z for height 0.5, 0, 0.5 or 0.7, 0, 0.5 couch
        #offset = np.array([2.3, 1.4, 0.5]) # stairs
        #offset = np.array([-1.5, -1, 0.3]) # carpet
        offset =  np.array([-1.5, -1, 0.1]) # carpet
        #offset = np.array([2.5, -6.5, 0.9]) # shelf (need to rotate camera) 
        #offset = np.array([1.3, 3.7, 0.5]) # dark room
        #offset = np.array([4.2, 0.5, 0.8]) # bicycle
        #offset = np.array([4.1, -5.3, 0.9]) # corner of a sofa 
        #offset = np.array([2.5, -5, 0.9]) # carpet
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
    print(1)
    make_box_space_readable()
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()

    if args.env_id in MS1_ENV_IDS:
        if args.control_mode is not None and not args.control_mode.startswith("base"):
            args.control_mode = "base_pd_joint_vel_arm_" + args.control_mode
    """ 
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
    """


    env = gym.make( # custom habitat2 background it overwrites the above env (comment it out if you want to use default background)
        "PickYCBInReplicaCAD-v0",
        #"PickMultiYCBInReplicaCAD-v0",
        obs_mode=args.obs_mode,

        render_camera_cfgs=dict(width=640, height=480),
        camera_cfgs=dict(hand_camera=dict(width=512, height=512)),
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        **args.env_kwargs)

    import sapien.core as sapien

    from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
    from mani_skill2.utils.registration import register_env
    from mani_skill2.utils.sapien_utils import set_render_material







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
    data_agent = DataAgent(env, args.env_id,args.control_mode, num_steps= 100)

    transformations = ["translation_Y"]
    num_episodes = 5
    for transformation in transformations:
        i=0
        for i in range(num_episodes):
            print("episode", i + 1)
            data_agent.run_episode('./test_data', f'data_{transformation}', transformation)
            env.reset(reconfigure=True)
        
    print("end of episodes")
    # Viewer


        





        # # -------------------------------------------------------------------------- #
        # # Post-process action
        # # -------------------------------------------------------------------------- #
        # if args.env_id in MS1_ENV_IDS:
        #     action_dict = dict(
        #         base=base_action,
        #         right_arm=ee_action,
        #         right_gripper=gripper_action,
        #         left_arm=np.zeros_like(ee_action),
        #         left_gripper=np.zeros_like(gripper_action),
        #     )
        #     action = env.agent.controller.from_action_dict(action_dict)
        # else:
        #     action_dict = dict(base=base_action, arm=ee_action, gripper=gripper_action)
        #     action = env.agent.controller.from_action_dict(action_dict)
        #
        # obs, reward, done, info = env.step(action)
        # if key == "g":
        #     go_to_pose(env, pose)
        # if key == "8":
        #     counter +=1
        #     img_array = obs['image']['hand_camera']['rgb']
        #     samples['intrinsic'].append(obs['camera_param']['hand_camera']['intrinsic_cv'])
        #     samples['extrinsic'].append(obs['camera_param']['hand_camera']['extrinsic_cv'])
        #     samples['image_rgb'].append(img_array)
        #     samples['depth'].append(obs['image']['hand_camera']['depth'])
        #     samples['name'].append(f"test_{counter:02}.png")
        #     samples['pose'].append(obs['extra']['tcp_pose'][0:3])
        #     roll, pitch, yaw = euler_from_quaternion(obs['extra']['tcp_pose'][3], obs['extra']['tcp_pose'][4],
        #                                              obs['extra']['tcp_pose'][5], obs['extra']['tcp_pose'][6])
        #     samples['euler_angles'].append([roll, pitch, yaw])
        #     print("image data recorded", samples['name'][-1])
        #
        # if key == "s":
        #     store_data(samples, "./test_data")
        #     print("data stored")
        # roll, pitch, yaw = euler_from_quaternion(obs['extra']['tcp_pose'][3], obs['extra']['tcp_pose'][4],
        #                                              obs['extra']['tcp_pose'][5], obs['extra']['tcp_pose'][6])
        # print("roll, pitch, yaw", roll, pitch, yaw)
        # print("tcp_pose", obs['extra']['tcp_pose'][0:3])

def go_to_pose(env, pose):
    action = np.zeros(6)
    action[0:3] = pose[0:3]
    action[3:6] = pose[3:6]
    obs, reward, done, info = env.step(action)
    return obs, reward, done, info

if __name__ == "__main__":
    main()
