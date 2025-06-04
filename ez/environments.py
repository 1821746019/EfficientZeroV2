import gymnasium as gym
from gymnasium.core import Wrapper
from gymnasium.wrappers import (
    RecordVideo,
    ResizeObservation,
    GrayScaleObservation,
    FrameStack,
)
import numpy as np
import collections
from typing import Any, Optional, SupportsFloat, Tuple, Dict


# Standard Atari wrappers (adapted from CleanRL / SB3)
class NoopResetEnv(Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        obs = None
        for _ in range(noops):
            obs, _, terminated, truncated, _ = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs) # Pass info back
        return obs, {} # Return empty info for consistency

    def step(self, action):
        return self.env.step(action)

class MaxAndSkipEnv(Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class EpisodicLifeEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated 
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            terminated = True # Treat loss of life as end of episode for training
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, _ = self.env.step(0) # no-op
            info = {} # Assuming no info from no-op step
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        return np.sign(float(reward))


def create_atari_env(
    game_name: str,
    seed: int,
    config: Any, # Hydra config for env parameters
    record_video_path: Optional[str] = None,
    frame_stack_axis: int = -1, # For FrameStack: -1 for (H,W,C*N), 0 for (N*C,H,W)
) -> gym.Env:

    if "NoFrameskip" not in game_name:
        env_id = game_name + "NoFrameskip-v4"
    else:
        env_id = game_name
    
    env = gym.make(env_id, obs_type="rgb") # Ensure RGB for grayscale conversion
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    # env.reset(seed=seed) # Gymnasium reset takes seed

    if record_video_path:
        env = RecordVideo(env, record_video_path, episode_trigger=lambda x: x % 50 == 0) # Record every 50th episode

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=config.env.frame_skip)
    
    if config.env.terminal_on_life_loss:
        env = EpisodicLifeEnv(env)

    # Resize and Grayscale
    if config.env.screen_size != env.observation_space.shape[0] or \
       config.env.screen_size != env.observation_space.shape[1]:
        env = ResizeObservation(env, (config.env.screen_size, config.env.screen_size))
    
    if config.env.grayscale:
        env = GrayScaleObservation(env, keep_dim=True) # keep_dim=True for (H,W,1)

    if config.env.clip_rewards:
        env = ClipRewardEnv(env)

    env = FrameStack(env, num_stack=config.env.frame_stack, lz4_compress=False) # axis default is -1
    
    # Final check on observation space for FrameStack
    # FrameStack changes obs space shape. If grayscale (H,W,1) -> (H,W,N_stack).
    # If RGB (H,W,3) -> (H,W,3*N_stack).
    # This matches common "channels_last" convention for CNNs in Flax.

    return env