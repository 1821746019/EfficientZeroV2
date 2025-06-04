import gymnasium as gym
from typing import Dict, Any, List
import cv2
import numpy as np
class FrameStack:
    """帧堆叠"""
    
    def __init__(self, num_frames: int):
        self.num_frames = num_frames
        self.frames = []
    
    def reset(self, frame):
        self.frames = [frame] * self.num_frames
        return self.get_observation()
    
    def step(self, frame):
        self.frames.append(frame)
        if len(self.frames) > self.num_frames:
            self.frames.pop(0)
        return self.get_observation()
    
    def get_observation(self):
        return np.stack(self.frames, axis=-1)

class AtariPreprocessor:
    """Atari预处理"""
    
    def __init__(self, height: int = 84, width: int = 84, grayscale: bool = True):
        self.height = height
        self.width = width
        self.grayscale = grayscale
    
    def process(self, frame):
        """处理帧"""
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height))
        if self.grayscale:
            frame = np.expand_dims(frame, axis=-1)
        return frame.astype(np.float32) / 255.0

class EZEnvironment:
    """EfficientZero环境包装器"""
    
    def __init__(self, env_name: str, config: Dict[str, Any]):
        import ale_py
        gym.register_envs(ale_py)
        self.env = gym.make(env_name)
        self.config = config
        
        # 设置预处理器
        if 'ALE' in env_name:  # Atari游戏
            self.preprocessor = AtariPreprocessor(
                height=config.get('height', 84),
                width=config.get('width', 84),
                grayscale=config.get('grayscale', True)
            )
            self.frame_stack = FrameStack(config.get('frame_stack', 4))
        else:
            self.preprocessor = None
            self.frame_stack = None
        
        self.action_space_size = self.env.action_space.n if hasattr(self.env.action_space, 'n') else self.env.action_space.shape[0]
        self.is_continuous = not hasattr(self.env.action_space, 'n')
    
    def reset(self):
        """重置环境"""
        obs, info = self.env.reset()
        
        if self.preprocessor:
            obs = self.preprocessor.process(obs)
        if self.frame_stack:
            obs = self.frame_stack.reset(obs)
        
        return obs, info
    
    def step(self, action):
        """执行动作"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.preprocessor:
            obs = self.preprocessor.process(obs)
        if self.frame_stack:
            obs = self.frame_stack.step(obs)
        
        done = terminated or truncated
        return obs, reward, done, info
    
    def close(self):
        self.env.close()

def make_env(env_name: str, config: Dict[str, Any]) -> EZEnvironment:
    """创建环境"""
    return EZEnvironment(env_name, config)

