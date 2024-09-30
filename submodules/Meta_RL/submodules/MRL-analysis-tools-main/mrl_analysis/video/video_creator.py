import sys
import numpy as np
import cv2
from collections import deque
from typing import List, Any, Dict, Callable
from gym.envs.mujoco.mujoco_env import MujocoEnv, BaseMujocoEnv
from pathlib import Path

from mrl_analysis.utility import interfaces
from .camera_wrapper import CameraWrapper


class VideoCreator():
    """A class for video rendering of policies.

    Use function ``create_video`` to render environment interactions and write
    the results to a video.

    You can modify ``fps`` to set the number of rendered frames per second.
    """
    def __init__(self) -> None:
        self.fps = 10.0
        self.buffer_size = 10

        # Context arrays
        self.observations: deque
        self.actions: deque
        self.rewards: deque
        self.next_observations: deque
        self.terminals: deque
        self.environment_steps = 0
        self.obs: np.ndarray = None

    def _init_buffers(self, encoder: interfaces.MdpEncoder, decoder: interfaces.MdpDecoder):
        """ Initialize buffers (used for context accumulation) """
        # Context data initialization
        self.observations = deque(
            [np.zeros((decoder.observation_dim)) for _ in range(encoder.context_size)],
            maxlen=encoder.context_size
        )
        self.actions = deque(
            [np.zeros((decoder.action_dim)) for _ in range(encoder.context_size)],
            maxlen=encoder.context_size
        )
        self.rewards = deque(
            [np.zeros((1)) for _ in range(encoder.context_size)],
            maxlen=encoder.context_size
        )
        self.next_observations = deque(
            [np.zeros((decoder.observation_dim)) for _ in range(encoder.context_size)],
            maxlen=encoder.context_size
        )
        self.terminals = deque(
            [np.zeros((1)) for _ in range(encoder.context_size)],
            maxlen=encoder.context_size
        )
        self.obs = None

    def create_video(
        self,
        results: Dict[str, Any],
        video_length: float = None,
        n_frames: int = None,
        save_as: str = "./videos/video.mp4",
        env_reset_interval: int = 250,
        width: int = None,
        height: int = None,
    ) -> None:
        """Create a video file from an environment, an encoder, and a policy

        Parameters
        ----------
        results : Dict[str, Any]
            Dictionary of (training) results, must include the following items:
                eval_env: ``interfaces.MetaEnv`` - Environment
                encoder: ``interfaces.MdpEncoder`` - Context encoder
                decoder: ``interfaces.MdpDecoder`` - Decoder (maps context + state + action to predictions)
                policy: ``interfaces.MetaRLPolicy`` - Policy for interaction, takes state + encoding as inputs
        video_length : float
            Length of the video in frames
        n_frames : int
            Number of frames in the video, can be provided *instead of video_length*
        save_as : str, optional
            Name of video file, by default "./videos/video.mp4"
        env_reset_interval : int, optional
            Reset interval for environment (steps), by default 250
        width : int, optional
            Width of rendered video (in pixels), by default None
        height : int, optional
            Height of rendered video (in pixels), by default None
        """

        if video_length is not None:
            n_frames = int(self.fps * video_length)
        if video_length is None:
            raise ValueError("You need to provide either 'video_length' of 'n_frames'.")

        # Create output path if it doesn't already exist
        Path(save_as).parent.mkdir(parents=True, exist_ok=True)

        # Read variables from results
        env: interfaces.MetaEnv = results['eval_env']
        encoder: interfaces.MdpEncoder = results['encoder']
        decoder: interfaces.MdpDecoder = results['decoder']
        policy: interfaces.MetaRLPolicy = results['policy']
        encoder.train(False)
        decoder.train(False)
        policy.train(False)

        # Transformation of each frame before it is written to video
        transform = lambda x: x

        # MujocoEnv support
        if isinstance(env.unwrapped, MujocoEnv) or isinstance(env.unwrapped, BaseMujocoEnv):
            env = CameraWrapper(env)
            transform = np.flipud

        self._init_buffers(encoder, decoder)
        collected_frames = 0
        self.environment_steps = 0

        # Collect frames and write them to the video file
        video: cv2.VideoWriter = None
        while collected_frames < n_frames:
            # Collect frames by simulation
            frames = self._collect_frames(
                env, encoder, policy, 
                min(n_frames - collected_frames, self.buffer_size), 
                env_reset_interval=env_reset_interval, w=width, h=height
            )
            collected_frames += len(frames)

            # Apply frame transformations
            frames = [transform(frame) for frame in frames]

            # Initialize VideoWriter (only once)
            if video is None:
                size = frames[0].shape
                video = cv2.VideoWriter(save_as, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (size[1], size[0]), True)
            
            # Write frames to video
            self._frames_to_video(video, frames, transform)
            sys.stdout.write(f"\r{collected_frames} of {n_frames} frames collected.")
        video.release()
        print("\nDone!")

    def _collect_frames(
        self,
        env: interfaces.MetaEnv,
        encoder: interfaces.MdpEncoder,
        policy: interfaces.MetaRLPolicy,
        n_frames: int,
        env_reset_interval: int = 250,
        w: int = None,  # width
        h: int = None,  # height
    ) -> List[np.ndarray]:
        """ Renders environment interactions and returns the frames """

        # Frame sampling by policy rollout
        frames = []
        if self.obs is None:
            self.obs, _ = env.reset()
        for t in range(n_frames):
            context = {
                'observations': np.array(self.observations),
                'actions': np.array(self.actions),
                'rewards': np.array(self.rewards),
                'next_observations': np.array(self.next_observations),
                'terminals': np.array(self.terminals)
            }
            latent = encoder.get_encoding(**context)
            
            action, _ = policy.get_action(self.obs, latent)
            next_obs, reward, terminal, _, info = env.step(action)

            kwargs = {}
            if w is not None: kwargs['width'] = w
            if h is not None: kwargs['height'] = h
            image = env.render('rgb_array', **kwargs)
            frames.append(image)

            self.observations.append(self.obs)
            self.actions.append(action)
            self.rewards.append(np.array([reward]))
            self.next_observations.append(next_obs)
            self.terminals.append(np.array([terminal]))
            self.obs = next_obs
            self.environment_steps += 1

            if self.environment_steps % env_reset_interval == 0 or terminal:
                env.sample_task()
                self.obs, _ = env.reset()

        return frames
    
    def _frames_to_video(self, video: cv2.VideoWriter, frames: List[np.ndarray], transform: Callable = None):
        """ Write collected frames to video file """
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            video.write(frame)