# ==============================================================================
# ez_jax/agents/ez_agent.py - EfficientZero智能体
# ==============================================================================
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple
import threading
import time
import logging
from functools import partial
from ez_jax.utils.format import get_num_devices
from ez_jax.envs.env_factory import make_env
from ez_jax.models.networks import EfficientZeroModel
from ez_jax.utils.format import DiscreteSupport, replicate_across_devices, unreplicate_from_devices,get_num_devices
from ez_jax.mcts.mctx_search import GumbelMuZeroFns
from ez_jax.config.config import get_atari_config
from ez_jax.training.trainer import TrajectoryBuffer
from ez_jax.data.trajectory import GameTrajectory
import optax
from typing import List

class EfficientZeroAgent:
    """EfficientZero智能体"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_devices = get_num_devices()
        # 日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 初始化模型和组件
        self._init_model()
        self._init_mcts()
        self._init_training()
        
        # 初始化环境
        self.env = make_env(config['env']['name'], config['env'])
        
        # 数据收集相关
        self.trajectory_buffer = TrajectoryBuffer(config['buffer']['max_size'])
        self.current_trajectory = None
        
        # 训练状态
        self.training_step = 0
        self.training_active = False
        
    
    def _init_model(self):
        """初始化模型"""
        self.model = EfficientZeroModel(
            num_channels=self.config['model']['num_channels'],
            num_blocks=self.config['model']['num_blocks'],
            action_space_size=self.config['env']['action_space_size'],
            value_support_size=self.config['model']['value_support_size'],
            reward_support_size=self.config['model']['reward_support_size'],
            is_continuous=self.config['env']['is_continuous']
        )
        
        # 初始化参数
        rng = jax.random.PRNGKey(self.config['seed'])
        dummy_obs = jnp.ones((1, *self.config['env']['obs_shape']))
        self.params = self.model.init(rng, dummy_obs, training=True, method=self.model.initial_inference)
        self.params = replicate_across_devices(self.params)
        
        self.logger.info("Parameter shapes after _init_model replication:")
        jax.tree_util.tree_map_with_path(
            lambda path, x: self.logger.info(f"Param {jax.tree_util.keystr(path)} shape: {x.shape}, ndim: {x.ndim}, dtype: {x.dtype}"),
            self.params
        )
        
        # 支持函数
        self.value_support = DiscreteSupport(
            vmin=self.config['model']['value_min'],
            vmax=self.config['model']['value_max'],
            num_atoms=self.config['model']['value_support_size']
        )
        self.reward_support = DiscreteSupport(
            vmin=self.config['model']['reward_min'],
            vmax=self.config['model']['reward_max'],
            num_atoms=self.config['model']['reward_support_size']
        )
    
    def _init_mcts(self):
        """初始化MCTS"""
        self.mcts = GumbelMuZeroFns(
            model=self.model,
            value_support=self.value_support,
            reward_support=self.reward_support,
            num_simulations=self.config['mcts']['num_simulations']
        )
    
    def _init_training(self):
        """初始化训练"""
        # 优化器
        schedule = optax.cosine_decay_schedule(
            init_value=self.config['optimizer']['lr'],
            decay_steps=self.config['training']['total_steps'],
            alpha=0.1
        )
        self.optimizer = optax.adam(learning_rate=schedule)
        self.opt_state = self.optimizer.init(self.params)
        self.opt_state = replicate_across_devices(self.opt_state)
    
    @partial(jax.pmap, axis_name='batch')
    def _inference_step(self, params, observations):
        """推理步骤"""
        states, values, policies = self.model.apply(
            params, observations, training=False, method=self.model.initial_inference
        )
        
        # 转换值支持到标量
        values_scalar = self.value_support.vector_to_scalar(values)
        
        return states, values_scalar, policies
    
    @partial(jax.pmap, axis_name='batch')
    def _mcts_search(self, params, rng_key, states, values, policies):
        """MCTS搜索"""
        search_output = self.mcts.search(params, rng_key, states, values, policies)
        return search_output
    
    def act(self, observation: np.ndarray, temperature: float = 1.0) -> Tuple[int, Dict[str, Any]]:
        """选择动作"""
        # 准备观察
        obs_batch = jnp.expand_dims(observation, 0)  # 添加批次维度
        obs_batch = jnp.repeat(obs_batch[None, :], self.num_devices, axis=0)  # 复制到所有设备
        
        # 初始推理
        rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
        rng_keys = jax.random.split(rng_key, self.num_devices)
        
        states, values, policies = self._inference_step(self.params, obs_batch)
        
        # MCTS搜索
        search_output = self._mcts_search(self.params, rng_keys, states, values, policies)
        
        # 提取结果（取第一个设备的结果）
        search_values = unreplicate_from_devices(search_output.search_values)[0]
        search_policies = unreplicate_from_devices(search_output.search_policies)[0]
        best_action = unreplicate_from_devices(search_output.best_actions)[0]
        
        # 根据温度调整策略
        if temperature > 0:
            probs = jax.nn.softmax(search_policies / temperature)
            action = np.random.choice(len(probs), p=np.array(probs))
        else:
            action = best_action
        
        # 返回信息
        info = {
            'search_value': float(search_values),
            'search_policy': np.array(search_policies),
            'value': float(unreplicate_from_devices(values)[0]),
            'policy': np.array(unreplicate_from_devices(policies)[0])
        }
        
        return int(action), info
    
    def collect_trajectory(self) -> GameTrajectory:
        """收集一条轨迹"""
        trajectory = GameTrajectory(max_length=self.config['data']['max_trajectory_length'])
        
        obs, info = self.env.reset()
        done = False
        step = 0
        
        while not done and step < self.config['data']['max_episode_steps']:
            # 选择动作
            action, act_info = self.act(obs, temperature=self.config['data']['temperature'])
            
            # 执行动作
            next_obs, reward, done, env_info = self.env.step(action)
            
            # 存储数据
            trajectory.append(
                obs=obs.copy(),
                action=action,
                reward=reward,
                value=act_info['value'],
                policy=act_info['policy'],
                search_value=act_info['search_value'],
                search_policy=act_info['search_policy'],
                done=done
            )
            
            obs = next_obs
            step += 1
        
        return trajectory
    
    @partial(jax.pmap, axis_name='batch')
    def _training_step(self, params, opt_state, batch):
        """训练步骤"""
        
        def loss_fn(params):
            observations, actions, value_targets, reward_targets, policy_targets = batch
            
            # 初始推理
            states, values, policies = self.model.apply(
                params, observations, training=True, method=self.model.initial_inference
            )
            
            # 初始损失
            value_loss = optax.softmax_cross_entropy(
                values, self.value_support.scalar_to_vector(value_targets[:, 0])
            ).mean()
            
            policy_loss = optax.softmax_cross_entropy(policies, policy_targets[:, 0]).mean()
            
            total_loss = value_loss + policy_loss
            
            # 展开损失
            unroll_steps = self.config['training']['unroll_steps']
            for step in range(unroll_steps):
                states, rewards, values, policies = self.model.apply(
                    params, states, actions[:, step], training=True,
                    method=self.model.recurrent_inference
                )
                
                reward_loss = optax.softmax_cross_entropy(
                    rewards, self.reward_support.scalar_to_vector(reward_targets[:, step])
                ).mean()
                
                value_loss = optax.softmax_cross_entropy(
                    values, self.value_support.scalar_to_vector(value_targets[:, step + 1])
                ).mean()
                
                policy_loss = optax.softmax_cross_entropy(policies, policy_targets[:, step + 1]).mean()
                
                total_loss += reward_loss + value_loss + policy_loss
            
            return total_loss
        
        # 计算梯度
        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = jax.lax.pmean(grads, axis_name='batch')  # 聚合梯度
        
        # 应用更新
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss
    
    def _prepare_batch(self, trajectories: List[GameTrajectory]) -> Tuple:
        """准备训练批次"""
        batch_data = {
            'observations': [],
            'actions': [],
            'value_targets': [],
            'reward_targets': [],
            'policy_targets': []
        }
        
        for traj in trajectories:
            for i in range(len(traj) - self.config['training']['unroll_steps']):
                sample = traj.get_training_sample(i, self.config['training']['unroll_steps'])
                if sample is not None:
                    for key, value in sample.items():
                        if key == 'observation':
                            batch_data['observations'].append(value)
                        else:
                            batch_data[key].append(value)
        
        # 转换为JAX数组并复制到设备
        batch = []
        for key in ['observations', 'actions', 'value_targets', 'reward_targets', 'policy_targets']:
            data = np.array(batch_data[key])
            # 重塑为适合pmap的形状
            if len(data) % self.num_devices != 0:
                # 填充到能被设备数整除
                pad_size = self.num_devices - (len(data) % self.num_devices)
                pad_shape = (pad_size,) + data.shape[1:]
                padding = np.zeros(pad_shape, dtype=data.dtype)
                data = np.concatenate([data, padding], axis=0)
            
            data = data.reshape(self.num_devices, -1, *data.shape[1:])
            batch.append(jnp.array(data))
        
        return tuple(batch)
    
    def train_step(self):
        """执行一步训练"""
        # 从缓冲区采样轨迹
        trajectories = self.trajectory_buffer.sample(self.config['training']['batch_size'])
        if len(trajectories) < self.config['training']['min_trajectories']:
            return None
        
        # 准备批次
        batch = self._prepare_batch(trajectories)
        
        # 训练步骤
        self.params, self.opt_state, loss = self._training_step(self.params, self.opt_state, batch)
        
        # 更新计数
        self.training_step += 1
        
        return unreplicate_from_devices(loss)
    
    def data_collection_loop(self):
        """数据收集循环（在后台线程运行）"""
        while self.training_active:
            trajectory = self.collect_trajectory()
            self.trajectory_buffer.add(trajectory)
            
            self.logger.info(f"Collected trajectory with {len(trajectory)} steps")
            
            # 控制收集速度
            time.sleep(0.1)
    
    def train(self, total_steps: int):
        """主训练循环"""
        self.training_active = True
        
        # 启动数据收集线程
        collection_threads = []
        for i in range(self.config['data']['num_collectors']):
            thread = threading.Thread(target=self.data_collection_loop)
            thread.start()
            collection_threads.append(thread)
        
        self.logger.info(f"Started {len(collection_threads)} data collection threads")
        
        # 等待收集一些初始数据
        while self.trajectory_buffer.size() < self.config['training']['min_trajectories']:
            time.sleep(1)
            self.logger.info(f"Waiting for data... Buffer size: {self.trajectory_buffer.size()}")
        
        # 训练循环
        for step in range(total_steps):
            loss = self.train_step()
            
            if loss is not None:
                self.logger.info(f"Training step {step}, Loss: {loss:.4f}, Buffer size: {self.trajectory_buffer.size()}")
            
            # 评估
            if step % self.config['training']['eval_interval'] == 0:
                self.evaluate()
            
            # 保存模型
            if step % self.config['training']['save_interval'] == 0:
                self.save_model(f"model_step_{step}.pkl")
        
        # 停止数据收集
        self.training_active = False
        for thread in collection_threads:
            thread.join()
        
        self.logger.info("Training completed")
    
    def evaluate(self, num_episodes: int = 10):
        """评估模型"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < self.config['eval']['max_episode_steps']:
                action, _ = self.act(obs, temperature=0.0)  # 贪婪策略
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                step += 1
            
            episode_rewards.append(total_reward)
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        self.logger.info(f"Evaluation: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
        
        return mean_reward, std_reward
    
    def save_model(self, filename: str):
        """保存模型"""
        import pickle
        
        model_data = {
            'params': unreplicate_from_devices(self.params),
            'config': self.config,
            'training_step': self.training_step
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filename}")
    
    def load_model(self, filename: str):
        """加载模型"""
        import pickle
        
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.params = replicate_across_devices(model_data['params'])
        self.training_step = model_data['training_step']
        
        self.logger.info("Parameter shapes after load_model replication:")
        jax.tree_util.tree_map_with_path(
            lambda path, x: self.logger.info(f"Param {jax.tree_util.keystr(path)} shape: {x.shape}, ndim: {x.ndim}, dtype: {x.dtype}"),
            self.params
        )
        
        self.logger.info(f"Model loaded from {filename}")
