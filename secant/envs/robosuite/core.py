import robosuite as suite
from typing import Union, List, Optional, Dict
from .adapter import RobosuiteAdapter
from robosuite.utils.mjcf_utils import ALL_TEXTURES, TEXTURES
from secant.wrappers import TimeLimit, SingleModality, FrameStack, RandomScene


__all__ = ["make_robosuite", "ALL_TASKS", "ALL_ROBOTS", "ALL_TEXTURES", "TEXTURES"]


ALL_TASKS = ["Door", "TwoArmPegInHole", "NutAssemblyRound", "TwoArmLift"]
ALL_ROBOTS = list(suite.ALL_ROBOTS)


def make_robosuite(
    task: str,
    robots: Union[str, List[str]] = "Panda",
    controller_configs: Optional[Union[Dict, List[Dict]]] = None,
    controller_types: Optional[Union[str, List[str]]] = "OSC_POSE",
    headless: bool = True,
    obs_modality: Optional[List[str]] = ["rgb"],
    render_camera: str = "frontview",
    control_freq: int = 20,
    episode_length: Optional[int] = 500,
    ignore_done: bool = False,
    hard_reset: bool = False,
    obs_cameras: Union[str, List[str]] = "agentview",
    channel_first: bool = True,
    image_height: int = 168,
    image_width: int = 168,
    camera_depths: bool = False,
    custom_reset_config: Optional[Union[Dict, str]] = None,
    mode: bool = "train",
    scene_id: Optional[int] = 0,
    reward_shaping: bool = True,
    verbose: bool = False,
    single_modality_obs: bool = True,
    frame_stack: Optional[int] = 3,
    **kwargs,
):
    assert channel_first, "we only support channel_first=True"
    assert task in ALL_TASKS, f"Task {task} does not exist"

    env = RobosuiteAdapter(
        task=task,
        robots=robots,
        controller_configs=controller_configs,
        controller_types=controller_types,
        headless=headless,
        obs_modality=obs_modality,
        render_camera=render_camera,
        control_freq=control_freq,
        episode_length=episode_length,
        ignore_done=ignore_done,
        hard_reset=hard_reset,
        obs_cameras=obs_cameras,
        channel_first=channel_first,
        image_height=image_height,
        image_width=image_width,
        camera_depths=camera_depths,
        custom_reset_config=custom_reset_config,
        mode=mode,
        scene_id=scene_id,
        reward_shaping=reward_shaping,
        verbose=verbose,
        **kwargs,
    )

    env = TimeLimit(env, max_episode_steps=episode_length)
    if single_modality_obs and len(obs_modality) == 1:
        env = SingleModality(env, obs_modality[0])
    if frame_stack and frame_stack > 0:
        env = FrameStack(env, frame_stack)
    return env

def make_robosuite_random_sample(
    task: str,
    robots: Union[str, List[str]] = "Panda",
    controller_configs: Optional[Union[Dict, List[Dict]]] = None,
    controller_types: Optional[Union[str, List[str]]] = "OSC_POSE",
    headless: bool = True,
    obs_modality: Optional[List[str]] = ["rgb"],
    render_camera: str = "frontview",
    control_freq: int = 20,
    episode_length: Optional[int] = 500,
    ignore_done: bool = False,
    hard_reset: bool = False,
    obs_cameras: Union[str, List[str]] = "agentview",
    channel_first: bool = True,
    image_height: int = 168,
    image_width: int = 168,
    camera_depths: bool = False,
    custom_reset_config: Optional[Union[Dict, str]] = None,
    mode: bool = "train",
    scene_ids: Optional[Union[int, List[int]]] = 0,
    reward_shaping: bool = True,
    verbose: bool = False,
    single_modality_obs: bool = True,
    frame_stack: Optional[int] = 3,
    **kwargs,
):
    if type(scene_ids) is not list:
        scene_ids = [scene_ids]
    kwargs.pop('scene_ids', None)
    envs = []
    for scene_id in scene_ids:
        env = make_robosuite(
            task=task,
            robots=robots,
            controller_configs=controller_configs,
            controller_types=controller_types,
            headless=headless,
            obs_modality=obs_modality,
            render_camera=render_camera,
            control_freq=control_freq,
            episode_length=episode_length,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            obs_cameras=obs_cameras,
            channel_first=channel_first,
            image_height=image_height,
            image_width=image_width,
            camera_depths=camera_depths,
            custom_reset_config=custom_reset_config,
            mode=mode,
            scene_id=scene_id,
            reward_shaping=reward_shaping,
            verbose=verbose,
            **kwargs,
        )
        envs.append(env)

    return RandomScene(envs)
    
