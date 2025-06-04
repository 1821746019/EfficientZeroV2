# ==============================================================================
# ez_jax/data/trajectory.py - 轨迹数据结构
# ==============================================================================
import numpy as np
from typing import List, Optional, Any
from dataclasses import dataclass

@dataclass
class GameTrajectory:
    """游戏轨迹数据结构"""
    observations: List[np.ndarray]
    actions: List[Any]
    rewards: List[float]
    values: List[float]
    policies: List[np.ndarray]
    search_values: List[float]
    search_policies: List[np.ndarray]
    dones: List[bool]
    
    def __init__(self, max_length: int = 1000):
        self.max_length = max_length
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.policies = []
        self.search_values = []
        self.search_policies = []
        self.dones = []
    
    def append(self, obs, action, reward, value, policy, search_value, search_policy, done):
        """添加一步数据"""
        if len(self.observations) >= self.max_length:
            # 移除最老的数据
            self.observations.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.policies.pop(0)
            self.search_values.pop(0)
            self.search_policies.pop(0)
            self.dones.pop(0)
        
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.policies.append(policy)
        self.search_values.append(search_value)
        self.search_policies.append(search_policy)
        self.dones.append(done)
    
    def get_training_sample(self, pos: int, unroll_steps: int = 5):
        """获取训练样本"""
        if pos + unroll_steps >= len(self.observations):
            return None
        
        obs = np.array(self.observations[pos])
        actions = np.array(self.actions[pos:pos+unroll_steps])
        
        # 计算值目标（n步回报）
        value_targets = []
        reward_targets = []
        policy_targets = []
        
        for i in range(unroll_steps + 1):
            if pos + i < len(self.rewards):
                # 计算n步回报
                n_step_return = 0
                discount = 1.0
                for j in range(min(5, len(self.rewards) - pos - i)):
                    n_step_return += discount * self.rewards[pos + i + j]
                    discount *= 0.997
                
                if pos + i + 5 < len(self.search_values):
                    n_step_return += discount * self.search_values[pos + i + 5]
                
                value_targets.append(n_step_return)
                if i < unroll_steps:
                    reward_targets.append(self.rewards[pos + i])
                    policy_targets.append(self.search_policies[pos + i])
        
        return {
            'observation': obs,
            'actions': actions,
            'value_targets': np.array(value_targets),
            'reward_targets': np.array(reward_targets),
            'policy_targets': np.array(policy_targets)
        }
    
    def __len__(self):
        return len(self.observations)
