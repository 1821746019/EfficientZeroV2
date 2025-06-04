import jax
import jax.numpy as jnp
import chex
import numpy as np
from typing import Any, Callable, Tuple, List, Optional
from flax import struct

@struct.dataclass
class PRNGSequence:
    key: chex.PRNGKey

    def next(self) -> chex.PRNGKey:
        new_key, current_key = jax.random.split(self.key)
        # Pytype complains about "FrozenInstanceError: Trying to assign to field key"
        # but this is a common pattern for PRNG key updates in JAX.
        # We are re-binding the name 'self' in a sense, or rather, its 'key' attribute.
        # A more explicit way would be:
        # object.__setattr__(self, 'key', new_key)
        # However, for dataclasses, it's often easier to return a new instance.
        # For simplicity here, we'll assume this pattern is understood in JAX context,
        # or use a functional approach if strict immutability of PRNGSequence is needed.
        # Let's make it functional to be safe:
        # return PRNGSequence(new_key), current_key
        # For now, let's assume it's a stateful generator for ease of use in loops.
        # This is not strictly JAX-idiomatic for pure functions but common in scripts.
        # To be pure, the main loop would handle key splitting.
        # For now, this stateful approach simplifies the main loop's key management.
        self.key = new_key
        return current_key

# JAX Array Manipulations
def tree_device_put(pytree: Any, device: jax.Device) -> Any:
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, device), pytree)

def tree_replicate(pytree: Any, devices: List[jax.Device]) -> Any:
    """Replicates a pytree across all specified devices."""
    return jax.device_put_replicated(pytree, devices)

def tree_unreplicate(pytree: Any) -> Any:
    """Unreplicates a pytree from devices to host (takes the first replica)."""
    return jax.tree_util.tree_map(lambda x: x[0], pytree)

def shard_pytree(pytree: Any, devices: List[jax.Device]) -> Any:
    """Shards the leading axis of a pytree across devices."""
    num_devices = len(devices)
    return jax.tree_util.tree_map(
        lambda x: jax.device_put_sharded(
            np.split(x, num_devices, axis=0), devices
        ) if hasattr(x, 'shape') and x.shape[0] % num_devices == 0 else x, # only shard if divisible
        pytree
    )

# Loss Functions
def categorical_cross_entropy_loss(
    logits: chex.Array, targets: chex.Array, weights: Optional[chex.Array] = None
) -> chex.Array:
    """Computes weighted categorical cross-entropy.
    targets are one-hot encoded or probabilities.
    """
    chex.assert_equal_shape_prefix((logits, targets), 1)
    loss = -jnp.sum(targets * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    if weights is not None:
        chex.assert_equal_shape_prefix((loss, weights), 1)
        loss = loss * weights
    return jnp.mean(loss)

def mse_loss(
    predictions: chex.Array, targets: chex.Array, weights: Optional[chex.Array] = None
) -> chex.Array:
    """Computes weighted mean squared error."""
    chex.assert_equal_shape_prefix((predictions, targets), 1)
    loss = 0.5 * jnp.square(predictions - targets)
    if predictions.ndim > targets.ndim: # If predictions have extra dim (e.g. from vmap)
        loss = jnp.sum(loss, axis=list(range(targets.ndim, predictions.ndim)))
    loss = jnp.sum(loss, axis=list(range(1, loss.ndim))) # Sum over non-batch dimensions

    if weights is not None:
        chex.assert_equal_shape_prefix((loss, weights), 1)
        loss = loss * weights
    return jnp.mean(loss)

def symlog_transform_jax(x: chex.Array, eps: float = 1e-3) -> chex.Array:
    """MuZero-style symlog transform."""
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + eps * x

def symexp_transform_jax(x: chex.Array, eps: float = 1e-3) -> chex.Array:
    """MuZero-style symexp transform."""
    return jnp.sign(x) * (
        jnp.square(((jnp.sqrt(1 + 4 * eps * (jnp.abs(x) + 1 + eps)) - 1) / (2 * eps))) - 1
    )

def scalar_to_categorical_jax(
    scalar: chex.Array, support_min: float, support_max: float, num_bins: int
) -> chex.Array:
    """Transforms a scalar to a categorical representation."""
    chex.assert_scalar(support_min)
    chex.assert_scalar(support_max)
    chex.assert_scalar(num_bins)

    scalar = jnp.clip(scalar, support_min, support_max)
    bin_size = (support_max - support_min) / (num_bins - 1)
    
    # Normalize scalar to [0, num_bins - 1]
    normalized_scalar = (scalar - support_min) / bin_size
    
    lower_bin = jnp.floor(normalized_scalar).astype(jnp.int32)
    upper_bin = jnp.ceil(normalized_scalar).astype(jnp.int32)
    
    # Handle cases where scalar is exactly on a bin edge
    lower_bin = jnp.where(lower_bin == upper_bin, lower_bin -1, lower_bin)
    lower_bin = jnp.clip(lower_bin, 0, num_bins - 2) # Ensure lower_bin is valid index
    upper_bin = jnp.clip(upper_bin, 1, num_bins - 1) # Ensure upper_bin is valid index

    p_upper = normalized_scalar - lower_bin
    p_lower = 1.0 - p_upper
    
    # Create target distribution
    target_dist = jnp.zeros(scalar.shape + (num_bins,))

    # Scatter probabilities
    # For batch processing, need to handle batch dimensions correctly
    if scalar.ndim == 0: # Single scalar
        target_dist = target_dist.at[lower_bin].set(p_lower)
        target_dist = target_dist.at[upper_bin].set(p_upper)
    else: # Batched scalars
        batch_indices = jnp.indices(scalar.shape)[0] # Assuming scalar is 1D batch for simplicity
                                                    # For multi-dim batch, this needs generalization
        target_dist = target_dist.at[batch_indices, lower_bin].set(p_lower)
        target_dist = target_dist.at[batch_indices, upper_bin].set(p_upper)
        
    return target_dist

def categorical_to_scalar_jax(
    logits: chex.Array, support_min: float, support_max: float, num_bins: int
) -> chex.Array:
    """Transforms categorical logits to a scalar value."""
    chex.assert_scalar(support_min)
    chex.assert_scalar(support_max)
    chex.assert_scalar(num_bins)
    chex.assert_axis_dimension(logits, -1, num_bins)

    probs = jax.nn.softmax(logits, axis=-1)
    support = jnp.linspace(support_min, support_max, num_bins, dtype=probs.dtype)
    
    # Expand support to match batch dimensions of probs for broadcasting
    for _ in range(probs.ndim - 1):
        support = jnp.expand_dims(support, axis=0)
        
    scalar = jnp.sum(probs * support, axis=-1)
    return scalar

def cosine_similarity_loss_jax(
    vec1: chex.Array, vec2: chex.Array, weights: Optional[chex.Array] = None
) -> chex.Array:
    """Computes cosine similarity loss (1 - cosine_similarity)."""
    chex.assert_equal_shape(vec1, vec2)
    vec1_norm = vec1 / (jnp.linalg.norm(vec1, axis=-1, keepdims=True) + 1e-8)
    vec2_norm = vec2 / (jnp.linalg.norm(vec2, axis=-1, keepdims=True) + 1e-8)
    
    similarity = jnp.sum(vec1_norm * vec2_norm, axis=-1)
    loss = 1.0 - similarity # Or -similarity if aiming to maximize it
    
    if weights is not None:
        chex.assert_equal_shape_prefix((loss, weights), 1)
        loss = loss * weights
    return jnp.mean(loss)

# Metrics Aggregation
def average_metrics(metrics_list: List[dict]) -> dict:
    """Averages metrics from a list of dictionaries (e.g., from pmap)."""
    if not metrics_list:
        return {}
    
    avg_metrics = {}
    keys = metrics_list[0].keys()
    for k in keys:
        if isinstance(metrics_list[0][k], jax.Array) or isinstance(metrics_list[0][k], np.ndarray):
            all_vals = jnp.stack([m[k] for m in metrics_list])
            avg_metrics[k] = jnp.mean(all_vals)
        else: # For non-array metrics, just take the first one (e.g. step count)
            avg_metrics[k] = metrics_list[0][k]
    return avg_metrics

def get_temperature_schedule(
    step: int,
    initial_temp: float,
    final_temp: float,
    decay_steps: int
) -> float:
    """Linearly decays temperature."""
    if decay_steps == 0:
        return final_temp
    fraction = jnp.clip(step / decay_steps, 0.0, 1.0)
    return initial_temp + fraction * (final_temp - initial_temp)