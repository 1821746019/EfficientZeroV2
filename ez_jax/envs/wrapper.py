# ==============================================================================
# ez_jax/envs/wrappers.py  
# ==============================================================================
import gymnasium as gym
import numpy as np
from typing import Tuple, Any

class AtariWrapper(gym.Wrapper):
    """Atari环境包装器"""
    
    def __init__(self, env, frame_skip: int = 4, frame_stack: int = 4):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.frames = []
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.frame_stack
        return self._get_obs(), info
    
    def step(self, action):
        total_reward = 0
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        self.frames.append(obs)
        if len(self.frames) > self.frame_stack:
            self.frames.pop(0)
            
        return self._get_obs(), total_reward, terminated, truncated, info
    
    def _get_obs(self):
        return np.stack(self.frames, axis=-1)
