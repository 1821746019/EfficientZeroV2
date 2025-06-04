import jax
import jax.numpy as jnp
import numpy as np
import chex
from typing import Any, NamedTuple, Tuple, Optional
from flax import struct

# Store transitions as a flat structure for easier JAX processing
@struct.dataclass
class Transition:
    observation: chex.Array # [H, W, C * Stack]
    action: chex.Array # Scalar
    reward: chex.Array # Scalar
    discount: chex.Array # Scalar (1.0 for non-terminal, 0.0 for terminal in actual env step)
    next_observation: chex.Array # [H, W, C * Stack]
    done: chex.Array # Scalar bool/float
    # For training EfficientZero
    policy_target: chex.Array # [Num_Actions] (from MCTS)
    value_target: chex.Array # Scalar (from MCTS/n-step)
    # Add other fields if needed, e.g. n-step returns, search_stats

@struct.dataclass
class ReplayBufferState:
    # Using NumPy arrays for storage on host, convert to JAX arrays when sampling for device
    # This is generally more memory efficient for large buffers.
    # Alternatively, could use GlobalDeviceArray for very large on-device buffers if available.
    observations: np.ndarray # [Capacity, H, W, C * Stack]
    actions: np.ndarray      # [Capacity]
    rewards: np.ndarray      # [Capacity]
    discounts: np.ndarray    # [Capacity]
    next_observations: np.ndarray # [Capacity, H, W, C * Stack]
    dones: np.ndarray        # [Capacity]
    policy_targets: np.ndarray # [Capacity, Num_Actions]
    value_targets: np.ndarray  # [Capacity]
    
    priorities: np.ndarray   # [Capacity] for PER
    
    current_idx: int = 0
    current_size: int = 0
    capacity: int = struct.field(pytree_node=False)
    obs_shape: Tuple[int, ...] = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False) # For policy_targets shape
    alpha: float = struct.field(pytree_node=False) # PER alpha
    beta: float = struct.field(pytree_node=False) # PER beta (can be scheduled)

def init_replay_buffer(
    capacity: int, obs_spec: Tuple, action_dim: int, config: Any
) -> ReplayBufferState:
    # obs_spec is the shape of a single observation AFTER stacking, e.g., (96, 96, 4)
    return ReplayBufferState(
        observations=np.zeros((capacity, *obs_spec), dtype=np.uint8), # Atari obs are uint8
        actions=np.zeros((capacity,), dtype=np.int32),
        rewards=np.zeros((capacity,), dtype=np.float32),
        discounts=np.zeros((capacity,), dtype=np.float32),
        next_observations=np.zeros((capacity, *obs_spec), dtype=np.uint8),
        dones=np.zeros((capacity,), dtype=np.bool_),
        policy_targets=np.zeros((capacity, action_dim), dtype=np.float32),
        value_targets=np.zeros((capacity,), dtype=np.float32),
        priorities=np.zeros((capacity,), dtype=np.float32),
        current_idx=0,
        current_size=0,
        capacity=capacity,
        obs_shape=obs_spec,
        action_dim=action_dim,
        alpha=config.replay_buffer.priority_alpha,
        beta=config.replay_buffer.priority_beta_initial # Beta can be scheduled
    )

def add_transitions_to_buffer(
    buffer_state: ReplayBufferState,
    # Expect a list of dictionaries, each dict is a full transition
    transitions: list[dict] # Each dict has keys like 'observation', 'action', etc.
) -> ReplayBufferState:
    
    num_to_add = len(transitions)
    if num_to_add == 0:
        return buffer_state

    # Determine max priority for new transitions
    max_priority = np.max(buffer_state.priorities) if buffer_state.current_size > 0 else 1.0

    new_idx = buffer_state.current_idx
    new_size = buffer_state.current_size

    for i in range(num_to_add):
        trans = transitions[i]
        buffer_state.observations[new_idx] = trans['observation']
        buffer_state.actions[new_idx] = trans['action']
        buffer_state.rewards[new_idx] = trans['reward']
        buffer_state.discounts[new_idx] = trans['discount']
        buffer_state.next_observations[new_idx] = trans['next_observation']
        buffer_state.dones[new_idx] = trans['done']
        buffer_state.policy_targets[new_idx] = trans['policy_target']
        buffer_state.value_targets[new_idx] = trans['value_target']
        buffer_state.priorities[new_idx] = max_priority # Initialize with max priority
        
        new_idx = (new_idx + 1) % buffer_state.capacity
        if new_size < buffer_state.capacity:
            new_size += 1
            
    return buffer_state.replace(current_idx=new_idx, current_size=new_size)


def sample_batch_from_buffer(
    buffer_state: ReplayBufferState,
    rng_key: chex.PRNGKey,
    batch_size: int, # Global batch size
    # n_step_return: int, # For calculating n-step returns if not stored directly
    # discount_gamma: float # For n-step returns
) -> Tuple[chex.Array, chex.ArrayTree, chex.Array]: # (indices, batch_pytree, is_weights)
    
    if buffer_state.current_size < batch_size: # Or some minimum size
        # Not enough samples, return empty or handle as error
        # For now, let's assume this is checked before calling
        raise ValueError("Not enough samples in replay buffer to sample a batch.")

    # Prioritized Experience Replay sampling
    probs = buffer_state.priorities[:buffer_state.current_size] ** buffer_state.alpha
    probs /= np.sum(probs)
    
    # Convert probs to JAX array for jax.random.choice
    # This happens on CPU, then indices are used to gather from NumPy arrays.
    # For extremely large buffers, sampling directly with NumPy might be faster.
    # jax_probs = jnp.array(probs)
    # indices = jax.random.choice(rng_key, buffer_state.current_size, shape=(batch_size,), p=jax_probs, replace=True)
    # Using numpy for sampling from numpy arrays directly
    indices = np.random.choice(buffer_state.current_size, size=batch_size, p=probs, replace=True)
    indices = jnp.array(indices) # Convert to JAX array for potential device transfer later

    # Importance sampling weights
    is_weights = (buffer_state.current_size * probs[indices.astype(np.int32)]) ** (-buffer_state.beta)
    is_weights /= np.max(is_weights) # Normalize
    is_weights = jnp.array(is_weights, dtype=jnp.float32)

    # Gather the batch data
    # This will create JAX arrays from the NumPy buffer slices
    # These arrays are still on host, to be moved to device by the caller
    batch_observations = jnp.array(buffer_state.observations[indices.astype(np.int32)])
    batch_actions = jnp.array(buffer_state.actions[indices.astype(np.int32)])
    batch_rewards = jnp.array(buffer_state.rewards[indices.astype(np.int32)])
    batch_discounts = jnp.array(buffer_state.discounts[indices.astype(np.int32)])
    batch_next_observations = jnp.array(buffer_state.next_observations[indices.astype(np.int32)])
    batch_dones = jnp.array(buffer_state.dones[indices.astype(np.int32)])
    batch_policy_targets = jnp.array(buffer_state.policy_targets[indices.astype(np.int32)])
    batch_value_targets = jnp.array(buffer_state.value_targets[indices.astype(np.int32)])

    # For EfficientZero, the value_target might be an n-step return.
    # If not pre-calculated and stored, it would be calculated here.
    # The original EZ calculates targets (value_prefix, value, policy) in BatchWorker.
    # Here, we assume policy_targets and value_targets are directly from MCTS/self-play.
    # N-step returns for value_targets could be computed here if needed:
    # value_targets = compute_n_step_returns(buffer_state, indices, n_step_return, discount_gamma)
    
    sampled_transitions = Transition(
        observation=batch_observations,
        action=batch_actions,
        reward=batch_rewards,
        discount=batch_discounts,
        next_observation=batch_next_observations,
        done=batch_dones,
        policy_target=batch_policy_targets,
        value_target=batch_value_targets
    )
    
    return indices, sampled_transitions, is_weights

def update_priorities_in_buffer(
    buffer_state: ReplayBufferState,
    indices: chex.Array, # JAX array of indices
    new_priorities: chex.Array # JAX array of new priorities
) -> ReplayBufferState:
    # Ensure priorities are positive
    new_priorities = np.abs(np.array(new_priorities)) + buffer_state.priority_epsilon
    
    # Update priorities in the NumPy array
    # Convert JAX arrays back to NumPy for indexing if they were on device
    np_indices = np.array(indices).astype(np.int32)
    np_new_priorities = np.array(new_priorities)
    
    buffer_state.priorities[np_indices] = np_new_priorities
    return buffer_state # Return modified buffer_state (priorities array is mutated)

def get_beta_for_schedule(current_step: int, config: Any) -> float:
    """Anneals beta for PER from initial to final over beta_steps."""
    fraction = min(float(current_step) / config.replay_buffer.priority_beta_steps, 1.0)
    beta = config.replay_buffer.priority_beta_initial + \
           fraction * (config.replay_buffer.priority_beta_final - config.replay_buffer.priority_beta_initial)
    return beta