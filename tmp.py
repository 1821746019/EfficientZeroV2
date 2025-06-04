# EfficientZero V2 JAX Implementation
# Based on the paper "EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data"

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import chex
import distrax
import rlax
import numpy as np
import collections
import gym
import tree as tree_util
from typing import Tuple, Optional, Any, NamedTuple
import time
import pickle
import os

# =============== Data Structures ===============

class ActorOutput(NamedTuple):
    action_tm1: jnp.ndarray
    reward: jnp.ndarray
    observation: jnp.ndarray
    first: jnp.ndarray
    last: jnp.ndarray

class AgentOutput(NamedTuple):
    state: jnp.ndarray
    logits: jnp.ndarray
    value_logits: jnp.ndarray
    value: jnp.ndarray
    reward_logits: jnp.ndarray
    reward: jnp.ndarray
    policy_mean: Optional[jnp.ndarray] = None
    policy_std: Optional[jnp.ndarray] = None

class Params(NamedTuple):
    encoder: Any
    prediction: Any
    transition: Any
    action_embedding: Optional[Any] = None

class GumbelTree(NamedTuple):
    state: jnp.ndarray
    logits: jnp.ndarray
    policy_mean: jnp.ndarray
    policy_std: jnp.ndarray
    reward_logits: jnp.ndarray
    reward: jnp.ndarray
    value_logits: jnp.ndarray
    value: jnp.ndarray
    q_values: jnp.ndarray
    visit_counts: jnp.ndarray
    depth: jnp.ndarray
    sampled_actions: jnp.ndarray
    simulation_rewards: jnp.ndarray

# =============== Utility Functions ===============

def scalar_to_two_hot(x: chex.Array, num_bins: int):
    """Categorical representation of real values."""
    max_val = (num_bins - 1) // 2
    x = jnp.clip(x, -max_val, max_val)
    x_low = jnp.floor(x).astype(jnp.int32)
    x_high = jnp.ceil(x).astype(jnp.int32)
    p_high = x - x_low
    p_low = 1. - p_high
    idx_low = x_low + max_val
    idx_high = x_high + max_val
    cat_low = jax.nn.one_hot(idx_low, num_bins) * p_low[..., None]
    cat_high = jax.nn.one_hot(idx_high, num_bins) * p_high[..., None]
    return cat_low + cat_high

def logits_to_scalar(logits: chex.Array):
    """Inverse of scalar_to_two_hot."""
    num_bins = logits.shape[-1]
    max_val = (num_bins - 1) // 2
    x = jnp.sum((jnp.arange(num_bins) - max_val) * jax.nn.softmax(logits), axis=-1)
    return x

def value_transform(x: chex.Array, epsilon: float = 1e-3):
    """Non-linear value transformation for variance reduction."""
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + epsilon * x

def inv_value_transform(x: chex.Array, epsilon: float = 1e-3):
    """Inverse of value_transform."""
    return jnp.sign(x) * (((jnp.sqrt(1 + 4 * epsilon * (jnp.abs(x) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)

def scale_gradient(g, scale: float):
    return g * scale + jax.lax.stop_gradient(g) * (1. - scale)

# =============== Network Architectures ===============

class ResidualBlock(hk.Module):
    def __init__(self, channels: int, name: str = "residual_block"):
        super().__init__(name=name)
        self.channels = channels
    
    def __call__(self, x):
        shortcut = x
        x = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.channels, kernel_shape=3, stride=1, padding='SAME', with_bias=False)(x)
        x = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.channels, kernel_shape=3, stride=1, padding='SAME', with_bias=False)(x)
        return shortcut + x

class StateEncoder(hk.Module):
    def __init__(self, channels: int, is_visual: bool = True, name: str = "state_encoder"):
        super().__init__(name=name)
        self.channels = channels
        self.is_visual = is_visual
    
    def __call__(self, observations: chex.Array) -> chex.Array:
        if self.is_visual:
            # For visual inputs (Atari, DMControl Vision)
            x = observations / 255.0
            x = hk.Conv2D(self.channels // 2, kernel_shape=3, stride=2, padding='SAME', with_bias=False)(x)
            x = ResidualBlock(self.channels // 2)(x)
            x = hk.Conv2D(self.channels, kernel_shape=3, stride=2, padding='SAME', with_bias=False)(x)
            x = ResidualBlock(self.channels)(x)
            x = hk.AvgPool(window_shape=(3, 3, 1), strides=(2, 2, 1), padding='SAME')(x)
            x = ResidualBlock(self.channels)(x)
            x = hk.AvgPool(window_shape=(3, 3, 1), strides=(2, 2, 1), padding='SAME')(x)
            x = ResidualBlock(self.channels)(x)
        else:
            # For proprioceptive inputs (DMControl Proprio)
            # Running mean normalization
            x = observations
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = hk.Linear(256)(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = jax.nn.relu(x)
            # Reshape to work with ResNet-style blocks
            batch_size = x.shape[0]
            x = x.reshape(batch_size, 1, 1, 256)
            x = ResidualBlock(256)(x)
            x = ResidualBlock(256)(x)
            x = ResidualBlock(256)(x)
        
        return x

class ActionEmbedding(hk.Module):
    def __init__(self, action_dim: int, embed_dim: int = 64, name: str = "action_embedding"):
        super().__init__(name=name)
        self.action_dim = action_dim
        self.embed_dim = embed_dim
    
    def __call__(self, actions: chex.Array) -> chex.Array:
        if len(actions.shape) == 0:  # Scalar action
            actions = actions[None]
        x = hk.Linear(self.embed_dim)(actions)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.relu(x)
        return x

class TransitionNetwork(hk.Module):
    def __init__(self, is_visual: bool = True, name: str = "transition"):
        super().__init__(name=name)
        self.is_visual = is_visual
    
    def __call__(self, state: chex.Array, action_embedding: chex.Array) -> chex.Array:
        channels = state.shape[-1]
        shortcut = state
        
        if self.is_visual:
            # Broadcast action embedding to spatial dimensions
            action_embed = action_embedding[None, None, :]
            action_embed = jnp.broadcast_to(action_embed, state.shape[:-1] + action_embedding.shape[-1:])
            x = jnp.concatenate([state, action_embed], axis=-1)
            x = hk.Conv2D(channels, kernel_shape=3, stride=1, padding='SAME', with_bias=False)(x)
            x = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)(x)
            x = jax.nn.relu(x + shortcut)
            x = ResidualBlock(channels)(x)
        else:
            # For proprioceptive inputs
            state_flat = state.reshape(-1)  # Flatten spatial dims
            x = jnp.concatenate([state_flat, action_embedding])
            x = hk.Linear(256)(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = jax.nn.relu(x)
            x = x.reshape(state.shape)  # Reshape back
            x = x + shortcut
            x = ResidualBlock(channels)(x)
        
        return x

class PredictionNetwork(hk.Module):
    def __init__(self, num_actions: int, num_bins: int, is_continuous: bool = False, 
                 is_visual: bool = True, output_init_scale: float = 0.0, name: str = "prediction"):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.num_bins = num_bins
        self.is_continuous = is_continuous
        self.is_visual = is_visual
        self.output_init_scale = output_init_scale
    
    def __call__(self, state: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, Optional[chex.Array], Optional[chex.Array]]:
        output_init = hk.initializers.VarianceScaling(scale=self.output_init_scale)
        
        if self.is_visual:
            # Reward head
            reward_head = hk.Sequential([
                hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Conv2D(16, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
                hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Flatten(-3),
                hk.Linear(32, with_bias=False),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Linear(self.num_bins, w_init=output_init),
            ])
            
            # Value head
            value_x = ResidualBlock(state.shape[-1])(state)
            value_head = hk.Sequential([
                hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Conv2D(16, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
                hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Flatten(-3),
                hk.Linear(32, with_bias=False),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Linear(self.num_bins, w_init=output_init),
            ])
            
            # Policy head
            policy_x = ResidualBlock(state.shape[-1])(state)
            if self.is_continuous:
                # Gaussian policy for continuous control
                policy_head = hk.Sequential([
                    hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
                    jax.nn.relu,
                    hk.Conv2D(16, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
                    hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
                    jax.nn.relu,
                    hk.Flatten(-3),
                    hk.Linear(32, with_bias=False),
                    hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                    jax.nn.relu,
                ])
                policy_features = policy_head(policy_x)
                policy_mean = hk.Linear(self.num_actions, w_init=output_init)(policy_features)
                policy_mean = 5.0 * jnp.tanh(policy_mean)  # Scale mean
                policy_std = hk.Linear(self.num_actions, w_init=output_init)(policy_features)
                policy_std = jax.nn.softplus(policy_std) + 1e-3  # Ensure positive std
                logits = None
            else:
                # Categorical policy for discrete control
                policy_head = hk.Sequential([
                    hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
                    jax.nn.relu,
                    hk.Conv2D(16, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
                    hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
                    jax.nn.relu,
                    hk.Flatten(-3),
                    hk.Linear(32, with_bias=False),
                    hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                    jax.nn.relu,
                    hk.Linear(self.num_actions, w_init=output_init),
                ])
                logits = policy_head(policy_x)
                policy_mean = policy_std = None
        else:
            # For proprioceptive inputs - flatten and use MLPs
            state_flat = state.reshape(-1)
            
            # Reward head
            reward_logits = hk.Sequential([
                hk.Linear(256),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Linear(self.num_bins, w_init=output_init),
            ])(state_flat)
            
            # Value head  
            value_logits = hk.Sequential([
                hk.Linear(256),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Linear(self.num_bins, w_init=output_init),
            ])(state_flat)
            
            # Policy head
            if self.is_continuous:
                policy_features = hk.Sequential([
                    hk.Linear(256),
                    hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                    jax.nn.relu,
                ])(state_flat)
                policy_mean = hk.Linear(self.num_actions, w_init=output_init)(policy_features)
                policy_mean = 5.0 * jnp.tanh(policy_mean)
                policy_std = hk.Linear(self.num_actions, w_init=output_init)(policy_features)
                policy_std = jax.nn.softplus(policy_std) + 1e-3
                logits = None
            else:
                logits = hk.Sequential([
                    hk.Linear(256),
                    hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                    jax.nn.relu,
                    hk.Linear(self.num_actions, w_init=output_init),
                ])(state_flat)
                policy_mean = policy_std = None
        
        if self.is_visual:
            reward_logits = reward_head(state)
            value_logits = value_head(value_x)
        
        return logits, reward_logits, value_logits, policy_mean, policy_std

# =============== Sampling-based Gumbel Search ===============

def gumbel_top_k_sample(rng_key: chex.PRNGKey, logits: chex.Array, k: int):
    """Sample top-k actions using Gumbel-Top-k trick."""
    gumbel_noise = jax.random.gumbel(rng_key, shape=logits.shape)
    noisy_logits = logits + gumbel_noise
    top_k_indices = jnp.argsort(noisy_logits)[-k:]
    return top_k_indices

def sample_actions_continuous(rng_key: chex.PRNGKey, policy_mean: chex.Array, 
                            policy_std: chex.Array, num_samples: int, exploration_ratio: float = 0.25):
    """Sample actions for continuous control with exploration."""
    action_dim = policy_mean.shape[-1]
    
    # Sample from current policy
    key1, key2 = jax.random.split(rng_key)
    num_policy_samples = int(num_samples * (1 - exploration_ratio))
    
    policy_samples = jax.random.normal(key1, (num_policy_samples, action_dim)) * policy_std + policy_mean
    
    # Sample from exploration distribution (flattened policy)
    num_explore_samples = num_samples - num_policy_samples
    explore_std = policy_std * 2.0  # Wider exploration
    explore_samples = jax.random.normal(key2, (num_explore_samples, action_dim)) * explore_std + policy_mean
    
    # Combine samples
    all_samples = jnp.concatenate([policy_samples, explore_samples], axis=0)
    
    # Clip to action bounds (assuming [-1, 1])
    all_samples = jnp.clip(all_samples, -1.0, 1.0)
    
    return all_samples

def sequential_halving(q_values: chex.Array, visit_counts: chex.Array, num_simulations: int):
    """Sequential Halving algorithm for action selection."""
    num_actions = q_values.shape[0]
    
    # Compute confidence intervals
    confidence_bonus = jnp.sqrt(jnp.log(visit_counts.sum() + 1) / jnp.maximum(visit_counts, 1))
    upper_confidence = q_values + confidence_bonus
    
    # Select best action
    best_action = jnp.argmax(upper_confidence)
    
    return best_action

# =============== EfficientZero V2 Agent ===============

class EfficientZeroV2Agent:
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, 
                 config: dict):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        
        # Determine input/output types
        self.is_visual = len(observation_space.shape) == 3
        self.is_continuous = isinstance(action_space, gym.spaces.Box)
        self.num_actions = action_space.shape[0] if self.is_continuous else action_space.n
        
        # Network functions
        self._encoder_fn = hk.without_apply_rng(hk.transform(
            lambda obs: StateEncoder(
                channels=config['channels'], 
                is_visual=self.is_visual
            )(obs)
        ))
        
        if self.is_continuous:
            self._action_embed_fn = hk.without_apply_rng(hk.transform(
                lambda actions: ActionEmbedding(
                    action_dim=self.num_actions,
                    embed_dim=config.get('action_embed_dim', 64)
                )(actions)
            ))
        
        self._prediction_fn = hk.without_apply_rng(hk.transform(
            lambda state: PredictionNetwork(
                num_actions=self.num_actions,
                num_bins=config['num_bins'],
                is_continuous=self.is_continuous,
                is_visual=self.is_visual,
                output_init_scale=config.get('output_init_scale', 0.0)
            )(state)
        ))
        
        self._transition_fn = hk.without_apply_rng(hk.transform(
            lambda state, action_embed: TransitionNetwork(
                is_visual=self.is_visual
            )(state, action_embed)
        ))
        
        # Search parameters
        self.num_simulations = config.get('num_simulations', 32)
        self.max_depth = config.get('max_depth', 5)
        self.discount = config.get('discount', 0.997)
        self.c1 = config.get('c1', 1.25)
        self.c2 = config.get('c2', 19625)
        
    def init_params(self, rng_key: chex.PRNGKey) -> Params:
        """Initialize network parameters."""
        enc_key, pred_key, trans_key, action_key = jax.random.split(rng_key, 4)
        
        # Dummy inputs
        dummy_obs = jnp.zeros((1,) + self.observation_space.shape)
        
        # Initialize encoder
        encoder_params = self._encoder_fn.init(enc_key, dummy_obs)
        dummy_state = self._encoder_fn.apply(encoder_params, dummy_obs)
        
        # Initialize prediction network
        prediction_params = self._prediction_fn.init(pred_key, dummy_state)
        
        # Initialize transition network
        if self.is_continuous:
            dummy_action = jnp.zeros(self.num_actions)
            action_embed_params = self._action_embed_fn.init(action_key, dummy_action)
            dummy_action_embed = self._action_embed_fn.apply(action_embed_params, dummy_action)
        else:
            dummy_action_embed = jax.nn.one_hot(0, self.num_actions)
            action_embed_params = None
            
        transition_params = self._transition_fn.init(trans_key, dummy_state[0], dummy_action_embed)
        
        return Params(
            encoder=encoder_params,
            prediction=prediction_params, 
            transition=transition_params,
            action_embedding=action_embed_params
        )
    
    def encode_state(self, params: Params, observation: chex.Array) -> chex.Array:
        """Encode observation to latent state."""
        return self._encoder_fn.apply(params.encoder, observation)
    
    def predict(self, params: Params, state: chex.Array) -> AgentOutput:
        """Make predictions from state."""
        logits, reward_logits, value_logits, policy_mean, policy_std = self._prediction_fn.apply(
            params.prediction, state
        )
        
        reward = inv_value_transform(logits_to_scalar(reward_logits))
        value = inv_value_transform(logits_to_scalar(value_logits))
        
        return AgentOutput(
            state=state,
            logits=logits,
            value_logits=value_logits,
            value=value,
            reward_logits=reward_logits,
            reward=reward,
            policy_mean=policy_mean,
            policy_std=policy_std
        )
    
    def transition(self, params: Params, state: chex.Array, action: chex.Array) -> chex.Array:
        """Predict next state given current state and action."""
        if self.is_continuous:
            action_embed = self._action_embed_fn.apply(params.action_embedding, action)
        else:
            action_embed = jax.nn.one_hot(action, self.num_actions)
        
        next_state = self._transition_fn.apply(params.transition, state, action_embed)
        next_state = scale_gradient(next_state, 0.5)  # Gradient scaling
        
        return next_state
    
    def gumbel_search(self, rng_key: chex.PRNGKey, params: Params, 
                     root_state: chex.Array, is_eval: bool = False) -> GumbelTree:
        """Perform sampling-based Gumbel search."""
        root_output = self.predict(params, root_state)
        
        if self.is_continuous:
            # Sample actions for continuous control
            key1, key2 = jax.random.split(rng_key)
            sampled_actions = sample_actions_continuous(
                key1, root_output.policy_mean, root_output.policy_std, 
                self.num_simulations, exploration_ratio=0.0 if is_eval else 0.25
            )
        else:
            # Sample actions for discrete control using Gumbel-Top-k
            sampled_actions = gumbel_top_k_sample(
                rng_key, root_output.logits, self.num_simulations
            )
        
        # Initialize search tree
        num_samples = sampled_actions.shape[0]
        q_values = jnp.zeros(num_samples)
        visit_counts = jnp.zeros(num_samples)
        simulation_rewards = jnp.zeros((num_samples, self.max_depth))
        
        # Perform simulations
        for sim_idx in range(self.num_simulations):
            action_idx = sim_idx % num_samples
            action = sampled_actions[action_idx]
            
            # Simulate trajectory
            trajectory_rewards = []
            current_state = root_state
            
            for depth in range(self.max_depth):
                # Transition to next state
                next_state = self.transition(params, current_state, action)
                next_output = self.predict(params, next_state)
                
                trajectory_rewards.append(next_output.reward)
                
                # Select next action (greedy for simplicity)
                if self.is_continuous:
                    action = next_output.policy_mean
                else:
                    action = jnp.argmax(next_output.logits)
                
                current_state = next_state
            
            # Calculate discounted return
            discounted_return = 0.0
            for i, reward in enumerate(trajectory_rewards):
                discounted_return += (self.discount ** i) * reward
            
            # Update statistics
            visit_counts = visit_counts.at[action_idx].add(1)
            old_q = q_values[action_idx] 
            new_q = (old_q * (visit_counts[action_idx] - 1) + discounted_return) / visit_counts[action_idx]
            q_values = q_values.at[action_idx].set(new_q)
            
            # Store simulation rewards
            sim_rewards = jnp.array(trajectory_rewards + [0.0] * (self.max_depth - len(trajectory_rewards)))
            simulation_rewards = simulation_rewards.at[action_idx].set(sim_rewards)
        
        return GumbelTree(
            state=root_state,
            logits=root_output.logits,
            policy_mean=root_output.policy_mean if self.is_continuous else None,
            policy_std=root_output.policy_std if self.is_continuous else None,
            reward_logits=root_output.reward_logits,
            reward=root_output.reward,
            value_logits=root_output.value_logits,
            value=root_output.value,
            q_values=q_values,
            visit_counts=visit_counts,
            depth=jnp.zeros(num_samples),
            sampled_actions=sampled_actions,
            simulation_rewards=simulation_rewards
        )
    
    def select_action(self, rng_key: chex.PRNGKey, tree: GumbelTree, 
                     temperature: float = 1.0, is_eval: bool = False) -> chex.Array:
        """Select action from search tree."""
        if is_eval:
            # Greedy action selection during evaluation
            best_idx = jnp.argmax(tree.q_values)
            return tree.sampled_actions[best_idx]
        
        # Temperature-based selection during training
        if self.is_continuous:
            # For continuous control, use weighted average based on Q-values
            weights = jax.nn.softmax(tree.q_values / temperature)
            action = jnp.sum(tree.sampled_actions * weights[:, None], axis=0)
            return action
        else:
            # For discrete control, use visit count based selection
            visit_probs = jax.nn.softmax(tree.visit_counts / temperature)
            return jax.random.choice(rng_key, tree.sampled_actions, p=visit_probs)
    
    def search_based_value_estimation(self, tree: GumbelTree) -> float:
        """Compute search-based value estimation (SVE)."""
        # Use mean of Q-values weighted by visit counts as value estimation
        weights = tree.visit_counts / jnp.maximum(tree.visit_counts.sum(), 1.0)
        sve_value = jnp.sum(tree.q_values * weights)
        return sve_value

# =============== Training Infrastructure ===============

class UniformReplayBuffer:
    def __init__(self, max_size: int, sequence_length: int):
        self.max_size = max_size
        self.sequence_length = sequence_length
        self.storage = []
        self.position = 0
        self.size = 0
    
    def add(self, trajectory: ActorOutput):
        """Add trajectory to buffer."""
        if len(self.storage) < self.max_size:
            self.storage.append(trajectory)
        else:
            self.storage[self.position] = trajectory
        
        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> Optional[ActorOutput]:
        """Sample batch of trajectories."""
        if self.size < batch_size:
            return None
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = [self.storage[i] for i in indices]
        
        # Stack trajectories
        return tree_util.map_structure(lambda *xs: jnp.stack(xs), *batch)

def create_loss_fn(agent: EfficientZeroV2Agent, config: dict):
    """Create loss function for training."""
    
    def loss_fn(params: Params, target_params: Params, trajectory: ActorOutput, 
                rng_key: chex.PRNGKey):
        unroll_steps = config['unroll_steps']
        td_steps = config['td_steps']
        discount = config['discount']
        
        # Encode initial state
        initial_state = agent.encode_state(params, trajectory.observation[0:1])
        root_output = agent.predict(params, initial_state[0])
        
        # Unroll model forward
        states = [initial_state[0]]
        predictions = [root_output]
        
        for t in range(unroll_steps):
            action = trajectory.action_tm1[t + 1]
            next_state = agent.transition(params, states[-1], action)
            next_pred = agent.predict(params, next_state)
            states.append(next_state)
            predictions.append(next_pred)
        
        # Generate targets using search
        search_targets = []
        value_targets = []
        
        for t in range(unroll_steps + 1):
            obs_t = trajectory.observation[t:t+1] 
            state_t = agent.encode_state(target_params, obs_t)[0]
            
            search_key = jax.random.split(rng_key, unroll_steps + 1)[t]
            tree = agent.gumbel_search(search_key, target_params, state_t, is_eval=False)
            
            # Policy target from search
            if agent.is_continuous:
                # Use best action from search as target
                best_idx = jnp.argmax(tree.q_values)
                policy_target = tree.sampled_actions[best_idx]
            else:
                # Use visit count distribution as target
                policy_target = jax.nn.softmax(tree.visit_counts)
            
            search_targets.append(policy_target)
            
            # Value target using mixed approach
            if t < config.get('early_training_steps', 1000):
                # Use TD target in early training
                bootstrap_value = predictions[min(t + td_steps, len(predictions) - 1)].value
                td_target = sum(
                    trajectory.reward[t + i] * (discount ** i) 
                    for i in range(min(td_steps, len(trajectory.reward) - t))
                ) + (discount ** min(td_steps, len(trajectory.reward) - t)) * bootstrap_value
                value_targets.append(td_target)
            else:
                # Use search-based value estimation
                sve_target = agent.search_based_value_estimation(tree)
                value_targets.append(sve_target)
        
        # Compute losses
        total_loss = 0.0
        logs = {}
        
        # Reward loss
        reward_targets = trajectory.reward[1:unroll_steps + 1]
        reward_targets = value_transform(reward_targets)
        reward_target_logits = jax.vmap(lambda x: scalar_to_two_hot(x, config['num_bins']))(reward_targets)
        
        reward_pred_logits = jnp.stack([pred.reward_logits for pred in predictions[1:unroll_steps + 1]])
        reward_loss = jnp.mean(jax.vmap(rlax.categorical_cross_entropy)(
            reward_target_logits, reward_pred_logits
        ))
        total_loss += reward_loss
        logs['reward_loss'] = reward_loss
        
        # Value loss
        value_targets_transformed = jnp.array([value_transform(vt) for vt in value_targets])
        value_target_logits = jax.vmap(lambda x: scalar_to_two_hot(x, config['num_bins']))(value_targets_transformed)
        
        value_pred_logits = jnp.stack([pred.value_logits for pred in predictions])
        value_loss = jnp.mean(jax.vmap(rlax.categorical_cross_entropy)(
            value_target_logits, value_pred_logits
        ))
        total_loss += config.get('value_coef', 0.25) * value_loss
        logs['value_loss'] = value_loss
        
        # Policy loss
        if agent.is_continuous:
            # MSE loss for continuous actions
            policy_targets = jnp.stack(search_targets)
            policy_preds = jnp.stack([pred.policy_mean for pred in predictions])
            policy_loss = jnp.mean((policy_targets - policy_preds) ** 2)
        else:
            # Cross-entropy loss for discrete actions  
            policy_targets = jnp.stack(search_targets)
            policy_logits = jnp.stack([pred.logits for pred in predictions])
            policy_loss = jnp.mean(jax.vmap(rlax.categorical_cross_entropy)(
                policy_targets, policy_logits
            ))
        
        total_loss += config.get('policy_coef', 1.0) * policy_loss
        logs['policy_loss'] = policy_loss
        logs['total_loss'] = total_loss
        
        return total_loss, logs
    
    return loss_fn

# =============== Training Loop ===============

class EfficientZeroV2Trainer:
    def __init__(self, agent: EfficientZeroV2Agent, config: dict):
        self.agent = agent
        self.config = config
        self.devices = jax.devices()  # Get available JAX devices
        num_devices = len(self.devices)
        print(f"INFO: EfficientZeroV2Trainer detected and will use {num_devices} JAX devices: {self.devices}")

        # Initialize parameters
        self.rng_key = jax.random.PRNGKey(config.get('seed', 42))
        init_key, self.rng_key = jax.random.split(self.rng_key)
        # Initialize parameters on CPU first before replicating
        unreplicated_params = agent.init_params(init_key)

        # Initialize optimizer
        learning_rate = optax.warmup_exponential_decay_schedule(
            init_value=0.0,
            peak_value=config.get('learning_rate', 3e-4),
            warmup_steps=config.get('warmup_steps', 1000),
            transition_steps=100000,
            decay_rate=0.1
        )

        self.optimizer = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=config.get('weight_decay', 1e-4)
        )
        # Initialize optimizer state on CPU first
        unreplicated_opt_state = self.optimizer.init(unreplicated_params)

        # Replicate params and opt_state to all available devices
        self.params = jax.device_put_replicated(unreplicated_params, self.devices)
        self.opt_state = jax.device_put_replicated(unreplicated_opt_state, self.devices)
        # Target parameters are also replicated
        self.target_params = self.params

        # Replay buffer
        self.replay_buffer = UniformReplayBuffer(
            max_size=config.get('replay_size', 100000),
            sequence_length=config.get('unroll_steps', 5) + config.get('td_steps', 5)
        )

        # Loss function
        self.loss_fn = create_loss_fn(agent, config)
        # pmap the _update_step function for parallel execution across devices
        self.update_fn = jax.pmap(
            self._update_step,
            axis_name='devices',  # Name for collective operations like pmean/psum
            devices=self.devices # Explicitly specify devices for pmap
        )

        # Tracking
        self.step_count = 0
        self.episode_returns = []

    def _update_step(self, params, target_params, opt_state, trajectory, rng_key):
        """Single training step, designed to be pmapped."""
        # params, target_params, opt_state are replicated by pmap.
        # trajectory and rng_key are sharded by pmap (one shard per device).

        (loss, logs), grads = jax.value_and_grad(
            self.loss_fn, has_aux=True
        )(params, target_params, trajectory, rng_key)  # rng_key is per-device here

        # Average gradients across all devices.
        grads = jax.lax.pmean(grads, axis_name='devices')
        # Average the main loss value across all devices.
        loss = jax.lax.pmean(loss, axis_name='devices')

        # Average scalar log values across devices. Non-scalar or non-JAX array logs are passed through.
        def average_scalar_logs(log_item):
            if isinstance(log_item, jnp.ndarray) and log_item.ndim == 0:
                return jax.lax.pmean(log_item, axis_name='devices')
            return log_item
        logs = jax.tree_util.tree_map(average_scalar_logs, logs)

        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Calculate norms based on the globally averaged gradients and updated parameters.
        # These will be identical across devices after this point.
        logs['grad_norm'] = optax.global_norm(grads)
        logs['param_norm'] = optax.global_norm(params)
        # Store the globally averaged loss. The 'total_loss' in logs from loss_fn is now averaged.
        # If 'total_loss' was already pmean-ed or is the pmean-ed 'loss' variable, ensure clarity.
        logs['total_loss_avg'] = loss # Explicitly note this is the averaged loss

        return params, opt_state, logs
    
    def collect_trajectory(self, env, max_steps: int = 1000) -> ActorOutput:
        """Collect a single trajectory from environment."""
        observations = []
        actions = []
        rewards = []
        firsts = []
        lasts = []
        
        obs = env.reset()
        observations.append(obs)
        firsts.append(True)
        lasts.append(False)
        actions.append(0)  # Dummy action for first step
        rewards.append(0.0)  # Dummy reward for first step
        
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            # Encode state and run search
            state = self.agent.encode_state(self.params, obs[None])[0]
            
            search_key, self.rng_key = jax.random.split(self.rng_key)
            tree = self.agent.gumbel_search(search_key, self.params, state, is_eval=False)
            
            # Select action
            action_key, self.rng_key = jax.random.split(self.rng_key)
            action = self.agent.select_action(action_key, tree, temperature=1.0, is_eval=False)
            
            # Execute action
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = int(action)
            
            obs, reward, done, info = env.step(action)
            
            observations.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            firsts.append(False)
            lasts.append(done)
            
            step_count += 1
        
        return ActorOutput(
            observation=jnp.array(observations),
            action_tm1=jnp.array(actions),
            reward=jnp.array(rewards),
            first=jnp.array(firsts),
            last=jnp.array(lasts)
        )
    
    def train_step(self):
        """Single training step with data sharding for pmap."""
        num_devices = len(self.devices)
        # Global batch size, must be divisible by num_devices
        global_batch_size = self.config.get('batch_size', 32)

        if global_batch_size % num_devices != 0:
            raise ValueError(
                f"Global batch size ({global_batch_size}) must be divisible by "
                f"the number of JAX devices ({num_devices}) for pmap."
            )
        
        device_batch_size = global_batch_size // num_devices

        # Sample a global batch from the replay buffer
        batch = self.replay_buffer.sample(global_batch_size)
        if batch is None:
            # print("Replay buffer does not have enough samples yet for a full global batch.")
            return None # Not enough samples for a full global batch

        # Shard the batch: each leaf in the ActorOutput PyTree needs to be reshaped.
        # Assumes ActorOutput fields are JAX arrays of shape (global_batch_size, ...)
        def shard_array(x: jnp.ndarray) -> jnp.ndarray:
            return x.reshape((num_devices, device_batch_size) + x.shape[1:])

        try:
            sharded_batch = tree_util.tree_map(shard_array, batch)
        except Exception as e:
            # Provide more context if sharding fails
            print(f"Error sharding batch for pmap: {e}")
            print("Details of the batch structure that failed to shard:")
            tree_util.tree_map_with_path(
                lambda path, x: print(f"Path: {path}, Shape: {x.shape}, Dtype: {x.dtype}"), batch
            )
            raise

        # Generate a unique RNG key for this update and then split it for each device
        update_key, self.rng_key = jax.random.split(self.rng_key)
        pmap_rng_keys = jax.random.split(update_key, num_devices)

        # Execute the pmapped update function.
        # self.params, self.opt_state, and self.target_params are already replicated across devices.
        # pmap will automatically pass the correct replicated/sharded view to each device.
        self.params, self.opt_state, logs = self.update_fn(
            self.params, self.target_params, self.opt_state, sharded_batch, pmap_rng_keys
        )

        # Logs returned from pmap are replicated (each device has the same pmean-ed log values).
        # We can take the logs from the first device for normal Python processing.
        processed_logs = jax.tree_util.tree_map(lambda x: x[0] if isinstance(x, jnp.ndarray) else x, logs)

        # Update target network periodically
        if self.step_count % self.config.get('target_update_interval', 200) == 0:
            # self.params are already the updated, replicated parameters from update_fn
            self.target_params = self.params

        self.step_count += 1
        return processed_logs
    
    def train(self, env, num_episodes: int):
        """Full training loop."""
        for episode in range(num_episodes):
            # Collect trajectory
            trajectory = self.collect_trajectory(env)
            episode_return = jnp.sum(trajectory.reward)
            self.episode_returns.append(float(episode_return))
            
            # Add to replay buffer
            self.replay_buffer.add(trajectory)
            
            # Training steps
            if self.replay_buffer.size >= self.config.get('min_replay_size', 1000):
                for _ in range(self.config.get('train_steps_per_episode', 1)):
                    logs = self.train_step()
                    if logs is not None and episode % 10 == 0:
                        print(f"Episode {episode}, Return: {episode_return:.2f}, "
                              f"Loss: {logs['total_loss_avg']:.4f}")
            
            if episode % 100 == 0:
                avg_return = np.mean(self.episode_returns[-100:])
                print(f"Episode {episode}, Average Return (last 100): {avg_return:.2f}")

# =============== Example Usage ===============

def create_default_config():
    """Create default configuration for EfficientZero V2."""
    return {
        # Network architecture
        'channels': 64,
        'num_bins': 601,
        'action_embed_dim': 64,
        'output_init_scale': 0.0,
        
        # Search parameters
        'num_simulations': 32,
        'max_depth': 5,
        'discount': 0.997,
        'c1': 1.25,
        'c2': 19625,
        
        # Training parameters
        'learning_rate': 3e-4,
        'warmup_steps': 1000,
        'weight_decay': 1e-4,
        'batch_size': 256,
        'unroll_steps': 5,
        'td_steps': 5,
        'value_coef': 0.25,
        'policy_coef': 1.0,
        'target_update_interval': 200,
        
        # Replay buffer
        'replay_size': 100000,
        'min_replay_size': 2000,
        'train_steps_per_episode': 1,
        
        # Training schedule
        'early_training_steps': 4000,
        'seed': 42
    }

def main():
    """Example training script."""
    # Create environment (replace with your desired environment)
    env = gym.make('CartPole-v1')  # Example discrete environment
    # env = gym.make('Pendulum-v1')  # Example continuous environment
    
    # Create config and agent
    config = create_default_config()
    agent = EfficientZeroV2Agent(env.observation_space, env.action_space, config)
    
    # Create trainer and train
    trainer = EfficientZeroV2Trainer(agent, config)
    trainer.train(env, num_episodes=1000)
    
    print("Training completed!")

if __name__ == "__main__":
    main()