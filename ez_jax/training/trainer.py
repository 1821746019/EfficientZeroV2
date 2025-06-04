# ==============================================================================
# ez_jax/training/trainer.py
# ==============================================================================
import optax
from functools import partial
import threading
import queue
from typing import Dict, Any
import logging
import jax
import jax.numpy as jnp
from ez_jax.models.networks import EfficientZeroModel
from ez_jax.utils.format import DiscreteSupport, replicate_across_devices, unreplicate_from_devices,get_num_devices
from ez_jax.mcts.mctx_search import GumbelMuZeroFns
from ez_jax.config.config import get_atari_config

class TrajectoryBuffer:
    """轨迹缓冲区（线程安全）"""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = queue.Queue(maxsize=max_size)
        self.lock = threading.Lock()
        
    def add(self, trajectory):
        """添加轨迹"""
        try:
            self.buffer.put(trajectory, block=False)
        except queue.Full:
            # 如果满了，移除最老的轨迹
            try:
                self.buffer.get(block=False)
                self.buffer.put(trajectory, block=False)
            except queue.Empty:
                pass
    
    def sample(self, batch_size: int):
        """采样批次"""
        batch = []
        for _ in range(min(batch_size, self.buffer.qsize())):
            try:
                trajectory = self.buffer.get(block=False)
                batch.append(trajectory)
            except queue.Empty:
                break
        return batch
    
    def size(self):
        return self.buffer.qsize()

class EfficientZeroTrainer:
    """EfficientZero训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_devices = get_num_devices()
        
        # 初始化模型
        self.model = EfficientZeroModel(
            num_channels=config['model']['num_channels'],
            num_blocks=config['model']['num_blocks'],
            action_space_size=config['env']['action_space_size'],
            value_support_size=config['model']['value_support_size'],
            reward_support_size=config['model']['reward_support_size'],
            is_continuous=config['env']['is_continuous']
        )
        
        # 初始化支持
        self.value_support = DiscreteSupport(
            vmin=config['model']['value_min'],
            vmax=config['model']['value_max'],
            num_atoms=config['model']['value_support_size']
        )
        self.reward_support = DiscreteSupport(
            vmin=config['model']['reward_min'],
            vmax=config['model']['reward_max'], 
            num_atoms=config['model']['reward_support_size']
        )
        
        # 初始化MCTS
        self.mcts = GumbelMuZeroFns(
            model=self.model,
            value_support=self.value_support,
            reward_support=self.reward_support,
            num_simulations=config['mcts']['num_simulations']
        )
        
        # 初始化优化器
        self.optimizer = optax.adam(
            learning_rate=config['optimizer']['lr'],
            b1=0.9,
            b2=0.999,
            eps=1e-8
        )
        
        # 初始化轨迹缓冲区
        self.trajectory_buffer = TrajectoryBuffer(config['buffer']['max_size'])
        
        # 初始化参数
        rng = jax.random.PRNGKey(config['seed'])
        dummy_obs = jnp.ones((1, *config['env']['obs_shape']))
        self.params = self.model.init(rng, dummy_obs, training=True)
        self.opt_state = self.optimizer.init(self.params)
        
        # 复制到所有设备
        self.params = replicate_across_devices(self.params)
        self.opt_state = replicate_across_devices(self.opt_state)
        
    @partial(jax.pmap, axis_name='batch')
    def loss_fn(self, params, batch):
        """计算损失函数"""
        observations, actions, targets = batch
        value_targets, reward_targets, policy_targets = targets
        
        # 初始推理
        states, values, policies = self.model.apply(
            params, observations, training=True, method=self.model.initial_inference
        )
        
        # 计算初始步的损失
        value_loss = optax.softmax_cross_entropy(
            values, self.value_support.scalar_to_vector(value_targets[:, 0])
        )
        policy_loss = optax.softmax_cross_entropy(policies, policy_targets[:, 0])
        
        total_loss = value_loss.mean() + policy_loss.mean()
        
        # 展开步数
        unroll_steps = self.config['training']['unroll_steps']
        for step in range(unroll_steps):
            # 递归推理
            states, rewards, values, policies = self.model.apply(
                params, states, actions[:, step], training=True,
                method=self.model.recurrent_inference
            )
            
            # 计算损失
            reward_loss = optax.softmax_cross_entropy(
                rewards, self.reward_support.scalar_to_vector(reward_targets[:, step])
            )
            value_loss = optax.softmax_cross_entropy(
                values, self.value_support.scalar_to_vector(value_targets[:, step + 1])
            )
            policy_loss = optax.softmax_cross_entropy(policies, policy_targets[:, step + 1])
            
            total_loss += (reward_loss.mean() + value_loss.mean() + policy_loss.mean())
        
        return total_loss
    
    @partial(jax.pmap, axis_name='batch')
    def update_step(self, params, opt_state, batch):
        """更新步骤"""
        loss, grads = jax.value_and_grad(self.loss_fn)(params, batch)
        
        # 聚合梯度
        grads = jax.lax.pmean(grads, axis_name='batch')
        
        # 应用梯度
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss
    
    @partial(jax.pmap, axis_name='batch')  
    def self_play_step(self, params, rng_key, observations):
        """自播放步骤"""
        # 初始推理
        states, values, policies = self.model.apply(
            params, observations, training=False, method=self.model.initial_inference
        )
        
        # MCTS搜索
        search_output = self.mcts.search(params, rng_key, states, values, policies)
        
        return search_output
    
    def collect_trajectories(self, num_trajectories: int):
        """收集轨迹"""
        # 这里应该实现环境交互和数据收集
        # 简化版本，实际需要根据具体环境实现
        pass
    
    def train_step(self):
        """训练步骤"""
        # 从缓冲区采样批次
        batch = self.trajectory_buffer.sample(self.config['training']['batch_size'])
        if len(batch) < self.config['training']['batch_size']:
            return None
        
        # 处理批次数据
        # ... 数据预处理逻辑 ...
        
        # 更新模型
        rng = jax.random.PRNGKey(0)  # 应该使用全局RNG状态
        self.params, self.opt_state, loss = self.update_step(
            self.params, self.opt_state, batch
        )
        
        return unreplicate_from_devices(loss)
    
    def train(self, num_steps: int):
        """主训练循环"""
        for step in range(num_steps):
            # 收集轨迹（在后台线程中）
            if step % self.config['training']['collect_interval'] == 0:
                threading.Thread(
                    target=self.collect_trajectories,
                    args=(self.config['data']['num_trajectories'],)
                ).start()
            
            # 训练步骤
            loss = self.train_step()
            if loss is not None:
                logging.info(f"Step {step}, Loss: {loss}")
            
            # 评估
            if step % self.config['training']['eval_interval'] == 0:
                self.evaluate()
    
    def evaluate(self):
        """评估模型"""
        # 实现评估逻辑
        pass

