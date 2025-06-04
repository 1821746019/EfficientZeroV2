import jax
import jax.numpy as jnp
from .utils import DiscreteSupportConfig, categorical_to_scalar, scalar_to_categorical, symlog, symexp

def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray, weights: jnp.ndarray = None) -> jnp.ndarray:
    """Computes weighted cross-entropy loss."""
    # logits: (B, num_classes), targets: (B, num_classes) (one-hot or soft targets)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(targets * log_probs, axis=-1) # (B,)
    if weights is not None:
        loss = loss * weights
    return loss.mean() # Mean over batch

def mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray, weights: jnp.ndarray = None) -> jnp.ndarray:
    """Computes weighted mean squared error."""
    # predictions: (B, ...), targets: (B, ...)
    loss = jnp.square(predictions - targets)
    # Reduce over non-batch dimensions if any
    if loss.ndim > 1:
        loss = jnp.mean(loss, axis=tuple(range(1, loss.ndim))) # (B,)
    if weights is not None:
        loss = loss * weights
    return loss.mean()

def symlog_mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray, weights: jnp.ndarray = None) -> jnp.ndarray:
    """ MSE loss in symlog space. """
    # predictions are network outputs (before symexp)
    # targets are true scalar values
    return mse_loss(predictions, symlog(targets), weights)


def cosine_similarity_loss(f1: jnp.ndarray, f2: jnp.ndarray, weights: jnp.ndarray = None) -> jnp.ndarray:
    """Computes weighted cosine similarity loss (negative similarity)."""
    # f1, f2: (B, D)
    f1_norm = f1 / (jnp.linalg.norm(f1, axis=-1, keepdims=True) + 1e-8)
    f2_norm = f2 / (jnp.linalg.norm(f2, axis=-1, keepdims=True) + 1e-8)
    similarity = jnp.sum(f1_norm * f2_norm, axis=-1) # (B,)
    loss = -similarity
    if weights is not None:
        loss = loss * weights
    return loss.mean()

# Example value loss for categorical representation
def value_loss_categorical(
    predicted_value_logits: jnp.ndarray, # (B, num_value_bins)
    target_value_scalar: jnp.ndarray,    # (B,)
    value_support_config: DiscreteSupportConfig,
    weights: jnp.ndarray = None
) -> jnp.ndarray:
    target_value_categorical = jax.vmap(scalar_to_categorical, in_axes=(0, None))(
        target_value_scalar, value_support_config
    ) # (B, num_value_bins)
    return cross_entropy_loss(predicted_value_logits, target_value_categorical, weights)

# Example policy loss
def policy_loss_discrete(
    predicted_policy_logits: jnp.ndarray, # (B, action_space_size)
    target_mcts_policy: jnp.ndarray,      # (B, action_space_size) (soft targets from MCTS)
    weights: jnp.ndarray = None
) -> jnp.ndarray:
    return cross_entropy_loss(predicted_policy_logits, target_mcts_policy, weights)