import jax
import jax.numpy as jnp
import chex
import mctx
from typing import Any, Callable, Tuple, Optional
from . import networks # Assuming networks.py for EfficientZeroNet
from . import utils    # For categorical transformations

# MCTX RootFnOutput and RecurrentFnOutput require specific fields.
# We'll wrap our network outputs to match these.

@chex.dataclass(frozen=True)
class MCTSStateEmbedding:
    """Embedding passed through MCTS, includes hidden state and optional LSTM state."""
    hidden_state: chex.Array  # From representation or dynamics net
    lstm_state: Optional[Tuple[chex.Array, chex.Array]] = None # For reward LSTM

def make_mcts_root_fn(
    model_apply_fn: Callable, # Bound model.apply method
    config: Any
) -> Callable[[chex.ArrayTree, chex.PRNGKey, chex.Array], mctx.RootFnOutput]:
    """
    Creates the root function for MCTX.
    Args:
        model_apply_fn: The `apply` method of the EfficientZeroNet model.
        config: Hydra configuration object.
    Returns:
        A function `(params, rng_key, observation_stack) -> mctx.RootFnOutput`.
    """
    def root_fn(params: chex.ArrayTree, rng_key: chex.PRNGKey, observation_stack: chex.Array) -> mctx.RootFnOutput:
        # model_apply_fn is already bound with variables like 'batch_stats' if using BatchNorm
        # The first argument to model_apply_fn is the params.
        # The call signature for EfficientZeroNet's initial_inference is (self, observations, train)
        # So, model_apply_fn({'params': params}, observation_stack, train=False)
        
        # We need to call the specific method for initial inference
        # This assumes EfficientZeroNet has a method like `initial_inference_apply`
        # or we pass method name to model_apply_fn.
        # Let's assume model_apply_fn can take a method_name argument.
        
        embedding, value_logits, policy_logits = model_apply_fn(
            {'params': params}, 
            observation_stack, 
            method=networks.EfficientZeroNet.initial_inference,
            train=False # MCTS is for inference/planning
        )
        # embedding is h_0
        # value_logits is V_logits(h_0)
        # policy_logits is P_logits(a|h_0)

        if config.train.use_categorical_value:
            value = utils.categorical_to_scalar_jax(
                value_logits, 
                config.train.value_support_min, 
                config.train.value_support_max, 
                config.train.value_support_bins
            )
            # MCTX expects raw value, not symlog transformed for its internal value.
            # If your network predicts symlog(value), you need to symexp it here.
            # Assuming value_logits directly lead to value after categorical_to_scalar.
        else:
            value = jnp.squeeze(value_logits, axis=-1)

        # MCTX embedding can be any pytree. We'll use our MCTSStateEmbedding.
        # For the root, LSTM state is typically initialized to zeros if used.
        initial_lstm_h = None
        initial_lstm_c = None
        if config.model.use_reward_lstm:
            batch_size = observation_stack.shape[0]
            lstm_hidden_size = config.model.reward_lstm_hidden_size
            initial_lstm_h = jnp.zeros((batch_size, lstm_hidden_size))
            initial_lstm_c = jnp.zeros((batch_size, lstm_hidden_size))
        
        mcts_embedding = MCTSStateEmbedding(
            hidden_state=embedding, 
            lstm_state=(initial_lstm_h, initial_lstm_c) if config.model.use_reward_lstm else None
        )
        
        return mctx.RootFnOutput(
            prior_logits=policy_logits,
            value=value,
            embedding=mcts_embedding
        )
    return root_fn

def make_mcts_recurrent_fn(
    model_apply_fn: Callable, # Bound model.apply method
    config: Any
) -> Callable[[chex.ArrayTree, chex.PRNGKey, chex.Array, MCTSStateEmbedding], 
              Tuple[mctx.RecurrentFnOutput, MCTSStateEmbedding]]:
    """
    Creates the recurrent function for MCTX.
    Args:
        model_apply_fn: The `apply` method of the EfficientZeroNet model.
        config: Hydra configuration object.
    Returns:
        A function `(params, rng_key, action, prev_embedding) -> 
                    (mctx.RecurrentFnOutput, next_mcts_embedding)`.
    """
    def recurrent_fn(
        params: chex.ArrayTree, 
        rng_key: chex.PRNGKey, 
        action: chex.Array, 
        prev_mcts_embedding: MCTSStateEmbedding
    ) -> Tuple[mctx.RecurrentFnOutput, MCTSStateEmbedding]:
        
        # prev_mcts_embedding contains hidden_state and lstm_state
        # Call the recurrent_inference method of the model
        (next_hidden_state, 
         reward_logits, 
         next_value_logits, 
         next_policy_logits, 
         next_lstm_state) = model_apply_fn(
            {'params': params},
            prev_mcts_embedding.hidden_state,
            action,
            prev_mcts_embedding.lstm_state,
            method=networks.EfficientZeroNet.recurrent_inference,
            train=False # MCTS is for inference/planning
        )

        if config.train.use_categorical_reward:
            reward = utils.categorical_to_scalar_jax(
                reward_logits,
                config.train.reward_support_min,
                config.train.reward_support_max,
                config.train.reward_support_bins
            )
        else:
            reward = jnp.squeeze(reward_logits, axis=-1)

        if config.train.use_categorical_value:
            next_value = utils.categorical_to_scalar_jax(
                next_value_logits,
                config.train.value_support_min,
                config.train.value_support_max,
                config.train.value_support_bins
            )
        else:
            next_value = jnp.squeeze(next_value_logits, axis=-1)
            
        # MCTX discount is for the transition: r_t + discount * V(s_{t+1})
        # This should be the game's discount factor.
        mctx_discount = jnp.full_like(reward, fill_value=config.train.discount)
        
        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=mctx_discount,
            prior_logits=next_policy_logits,
            value=next_value
        )
        
        next_mcts_embedding = MCTSStateEmbedding(
            hidden_state=next_hidden_state,
            lstm_state=next_lstm_state
        )
        
        return recurrent_fn_output, next_mcts_embedding
    return recurrent_fn

def get_mcts_q_transform(config: Any) -> Callable:
    """Configures and returns the Q-transform function for MCTX."""
    q_config = config.mcts.q_transform
    return mctx.qtransform_completed_by_mix_value_config(
        value_scale=q_config.value_scale,
        maxvisit_init=q_config.maxvisit_init,
        rescale_values=q_config.rescale_values,
        use_mixed_value=q_config.use_mixed_value
    )

def run_mcts(
    params: chex.ArrayTree,
    rng_key: chex.PRNGKey,
    observation_stack: chex.Array,
    model_apply_fn: Callable, # Bound model.apply
    config: Any,
    num_simulations: Optional[int] = None, # Allow override for eval
    temperature: Optional[float] = None # Allow override for eval
) -> mctx.PolicyOutput:
    """
    Runs MCTS search using MCTX's Gumbel MuZero policy.
    """
    if num_simulations is None:
        num_simulations = config.mcts.num_simulations
    if temperature is None:
        # Temperature for Gumbel noise scaling, not for final action selection sampling
        # MCTX Gumbel policy uses gumbel_scale for noise, not a temperature for policy smoothing during search.
        # The temperature here is for the final action selection from visit counts, which Gumbel MuZero policy in mctx handles differently.
        # Gumbel MuZero policy selects action based on max(gumbel + logits + q_values_from_completed_visits).
        # The original EZ code's temperature was for sampling from visit counts.
        # For Gumbel MuZero, the action selection is more deterministic after search.
        # We might need a separate temperature for sampling if not using the argmax from Gumbel.
        # For now, let's assume gumbel_scale handles the exploration aspect.
        gumbel_scale = config.mcts.gumbel_scale
    else: # If temperature is passed (e.g. for eval), it might mean gumbel_scale=0 for deterministic
        gumbel_scale = temperature # A bit of a misnomer, but if temp=0 means deterministic for eval

    root_fn = make_mcts_root_fn(model_apply_fn, config)
    recurrent_fn = make_mcts_recurrent_fn(model_apply_fn, config)
    q_transform_fn = get_mcts_q_transform(config)

    # Invalid actions mask (all actions are initially valid for Atari in MCTS, env handles termination)
    # MCTX expects 0 for valid, 1 for invalid.
    batch_size = observation_stack.shape[0]
    num_actions = config.env.action_space_size
    invalid_actions_mask = jnp.zeros((batch_size, num_actions), dtype=jnp.int32)

    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root_fn(params, rng_key, observation_stack), # Pass params and rng_key to root_fn
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_num_considered_actions=config.mcts.max_num_considered_actions,
        qtransform=q_transform_fn,
        invalid_actions=invalid_actions_mask,
        gumbel_scale=gumbel_scale,
        # loop_fn can be jax.lax.fori_loop (default) or hk.fori_loop if using Haiku
    )
    return policy_output