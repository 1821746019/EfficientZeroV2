import flax.linen as nn
from typing import Sequence
import jax.numpy as jnp
class ResidualBlock(nn.Module):
    """残差块"""
    channels: int
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        residual = x
        
        x = nn.Conv(self.channels, (3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        x = nn.Conv(self.channels, (3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        
        x = x + residual
        x = nn.relu(x)
        return x

class DownSample(nn.Module):
    """下采样网络"""
    out_channels: int
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # 第一次下采样
        x = nn.Conv(self.out_channels // 2, (3, 3), strides=(2, 2), 
                   padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # 残差块
        x = ResidualBlock(self.out_channels // 2)(x, training)
        
        # 第二次下采样
        residual = nn.Conv(self.out_channels, (3, 3), strides=(2, 2), 
                          padding='SAME', use_bias=False)(x)
        x = nn.Conv(self.out_channels, (3, 3), strides=(2, 2), 
                   padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = x + residual
        x = nn.relu(x)
        
        # 更多残差块和池化
        x = ResidualBlock(self.out_channels)(x, training)
        x = nn.avg_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        x = ResidualBlock(self.out_channels)(x, training)
        x = nn.avg_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        
        return x

class RepresentationNetwork(nn.Module):
    """表示网络：将观察编码为隐状态"""
    num_channels: int = 64
    num_blocks: int = 1
    downsample: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        if self.downsample:
            x = DownSample(self.num_channels)(x, training)
        else:
            x = nn.Conv(self.num_channels, (3, 3), padding='SAME', use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
        
        for _ in range(self.num_blocks):
            x = ResidualBlock(self.num_channels)(x, training)
        
        return x

class DynamicsNetwork(nn.Module):
    """动力学网络：预测下一个隐状态"""
    num_channels: int = 64
    num_blocks: int = 1
    action_space_size: int = 18
    
    @nn.compact
    def __call__(self, state, action, training: bool = True):
        batch_size = state.shape[0]
        state_shape = state.shape[1:]  # (H, W, C)
        
        # 动作编码
        if len(action.shape) == 1:  # 离散动作
            action_plane = jnp.ones((batch_size, *state_shape[:2], 1))
            action_plane = action_plane * (action[:, None, None, None] / self.action_space_size)
        else:  # 连续动作
            action_plane = jnp.broadcast_to(
                action[:, None, None, :], 
                (batch_size, *state_shape[:2], action.shape[-1])
            )
        
        # 拼接状态和动作
        x = jnp.concatenate([state, action_plane], axis=-1)
        
        x = nn.Conv(self.num_channels, (3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        
        x = x + state  # 残差连接
        x = nn.relu(x)
        
        for _ in range(self.num_blocks):
            x = ResidualBlock(self.num_channels)(x, training)
        
        return x

class ValuePolicyNetwork(nn.Module):
    """价值策略网络"""
    num_channels: int = 64
    reduced_channels: int = 16
    num_blocks: int = 1
    value_support_size: int = 601
    action_space_size: int = 18
    fc_hidden_size: int = 256
    is_continuous: bool = False
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # 残差块
        for _ in range(self.num_blocks):
            x = ResidualBlock(self.num_channels)(x, training)
        
        # 价值头
        value = nn.Conv(self.reduced_channels, (1, 1))(x)
        value = nn.BatchNorm(use_running_average=not training)(value)
        value = nn.relu(value)
        value = value.reshape(value.shape[0], -1)  # 展平
        value = nn.Dense(self.fc_hidden_size)(value)
        value = nn.relu(value)
        value = nn.Dense(self.value_support_size)(value)
        
        # 策略头
        policy = nn.Conv(self.reduced_channels, (1, 1))(x)
        policy = nn.BatchNorm(use_running_average=not training)(policy)
        policy = nn.relu(policy)
        policy = policy.reshape(policy.shape[0], -1)  # 展平
        policy = nn.Dense(self.fc_hidden_size)(policy)
        policy = nn.relu(policy)
        
        if self.is_continuous:
            # 连续动作：输出均值和标准差
            policy = nn.Dense(self.action_space_size * 2)(policy)
            mu, log_std = jnp.split(policy, 2, axis=-1)
            mu = jnp.tanh(mu)  # 限制均值范围
            std = jnp.exp(jnp.clip(log_std, -5, 2)) + 0.1  # 限制标准差范围
            policy = jnp.concatenate([mu, std], axis=-1)
        else:
            # 离散动作：输出logits
            policy = nn.Dense(self.action_space_size)(policy)
        
        return value, policy

class RewardNetwork(nn.Module):
    """奖励预测网络"""
    num_channels: int = 64
    reduced_channels: int = 16
    reward_support_size: int = 601
    fc_hidden_size: int = 256
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Conv(self.reduced_channels, (1, 1))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = x.reshape(x.shape[0], -1)  # 展平
        x = nn.Dense(self.fc_hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.reward_support_size)(x)
        return x

class EfficientZeroModel(nn.Module):
    """EfficientZero主模型"""
    num_channels: int = 64
    num_blocks: int = 1
    reduced_channels: int = 16
    action_space_size: int = 18
    value_support_size: int = 601
    reward_support_size: int = 601
    downsample: bool = True
    is_continuous: bool = False
    
    def setup(self):
        self.representation = RepresentationNetwork(
            num_channels=self.num_channels,
            num_blocks=self.num_blocks,
            downsample=self.downsample
        )
        
        self.dynamics = DynamicsNetwork(
            num_channels=self.num_channels,
            num_blocks=self.num_blocks,
            action_space_size=self.action_space_size
        )
        
        self.value_policy = ValuePolicyNetwork(
            num_channels=self.num_channels,
            reduced_channels=self.reduced_channels,
            num_blocks=self.num_blocks,
            value_support_size=self.value_support_size,
            action_space_size=self.action_space_size,
            is_continuous=self.is_continuous
        )
        
        self.reward = RewardNetwork(
            num_channels=self.num_channels,
            reduced_channels=self.reduced_channels,
            reward_support_size=self.reward_support_size
        )
    
    def initial_inference(self, observation, training: bool = True):
        """初始推理：从观察得到状态、价值和策略"""
        state = self.representation(observation, training)
        value, policy = self.value_policy(state, training)
        return state, value, policy
    
    def recurrent_inference(self, state, action, training: bool = True):
        """递归推理：从状态和动作得到下一状态、奖励、价值和策略"""
        next_state = self.dynamics(state, action, training)
        reward = self.reward(next_state, training)
        value, policy = self.value_policy(next_state, training)
        return next_state, reward, value, policy
