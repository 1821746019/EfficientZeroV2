import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import chex
from typing import Any, Callable, Tuple, Dict, Optional
from functools import partial

from . import networks
from . import utils
from . import mcts_config as mcts_cfg # Renamed to avoid conflict
from .replay_buffer import Transition # For typing batch data

# Define TrainState for the agent
class TrainState(train_state.TrainState):
    target_params: chex.ArrayTree
    # Add batch_stats for BatchNorm if not using TrainState's built-in support
    # model_state: chex.ArrayTree # For batch norm, dropout etc.
    # If using Flax's nn.BatchNorm, its state is stored in `batch_stats`
    # which TrainState can manage if `apply_fn` is correctly used with `mutable=['batch_stats']`.
    # For simplicity, let's assume model_apply_fn handles this.
    # We might need to explicitly pass and update batch_stats if not.
    # Let's assume for now that model_apply_fn will be called with mutable=['batch_stats']
    # and the updated batch_stats will be part of the new TrainState.
    batch_stats: chex.ArrayTree # For BatchNorm moving averages
    self_play_params: chex.ArrayTree # Periodically updated for actors
    rng_key: chex.PRNGKey # For dropout, etc., during training step

def create_train_state(
    rng_key: chex.PRNGKey, model: nn.Module, optimizer: optax.GradientTransformation,
    dummy_observation: chex.Array, dummy_action_sequence: Optional[chex.Array]
) -> TrainState:
    
    params_key, dropout_key, self_play_key = jax.random.split(rng_key, 3)
    
    variables = model.init(
        {'params': params_key, 'dropout': dropout_key}, 
        dummy_observation, 
        dummy_action_sequence, # Can be None if only initial_inference is used for init
        train=False # Use train=False for init to avoid issues with mutable state if any
    )
    params = variables['params']
    batch_stats = variables.get('batch_stats', {}) # Empty dict if no BatchNorm

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        target_params=params, # Initialize target params same as online params
        batch_stats=batch_stats,
        self_play_params=params, # Initialize self-play params
        rng_key=rng_key # Store a base rng_key, to be split further
    )

class EfficientZeroAgent:
    def __init__(self, config: Any, dummy_observation_spec: Tuple, action_dim: int):
        self.config = config
        self.model = networks.EfficientZeroNet(config=config)
        self.optimizer = self._build_optimizer()
        self.action_dim = action_dim # For policy head size

        # For initializing the model and train_state
        self.dummy_observation = jnp.zeros((1, *dummy_observation_spec), dtype=jnp.float32)
        if self.config.train.num_unroll_steps > 0:
            # Action can be int for discrete, or float array for continuous
            # Assuming discrete action space for Atari
            dummy_action_dtype = jnp.int32
            self.dummy_action_sequence = jnp.zeros(
                (1, self.config.train.num_unroll_steps), dtype=dummy_action_dtype
            )
        else:
            self.dummy_action_sequence = None


    def _build_optimizer(self) -> optax.GradientTransformation:
        cfg_opt = self.config.optimizer
        
        if cfg_opt.lr_schedule.name == "warmup_cosine_decay":
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0, # Start from 0 for warmup
                peak_value=cfg_opt.lr_schedule.peak_value,
                warmup_steps=cfg_opt.lr_schedule.warmup_steps,
                decay_steps=self.config.total_training_steps - cfg_opt.lr_schedule.warmup_steps,
                end_value=cfg_opt.lr_schedule.end_value
            )
        elif cfg_opt.lr_schedule.name == "constant":
            schedule = optax.constant_schedule(cfg_opt.learning_rate)
        else:
            raise ValueError(f"Unsupported learning rate schedule: {cfg_opt.lr_schedule.name}")

        if cfg_opt.name == "adamw":
            optimizer = optax.adamw(
                learning_rate=schedule,
                b1=cfg_opt.b1,
                b2=cfg_opt.b2,
                weight_decay=cfg_opt.weight_decay
            )
        elif cfg_opt.name == "adam":
            optimizer = optax.adam(
                learning_rate=schedule,
                b1=cfg_opt.b1,
                b2=cfg_opt.b2,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {cfg_opt.name}")

        if self.config.train.max_grad_norm > 0:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.train.max_grad_norm),
                optimizer
            )
        return optimizer

    def initial_train_state(self, rng_key: chex.PRNGKey) -> TrainState:
        return create_train_state(
            rng_key, self.model, self.optimizer, 
            self.dummy_observation, self.dummy_action_sequence
        )

    @partial(jax.jit, static_argnums=(0, 6)) # Jit the loss function
    def _loss_fn(
        self, params: chex.ArrayTree, target_params: chex.ArrayTree,
        batch_stats: chex.ArrayTree, # Current batch_stats for online model
        batch: Transition, # Sampled batch of transitions (unrolled)
        is_weights: chex.Array, # Importance sampling weights
        rng_key: chex.PRNGKey, # For dropout or other stochasticity in model
        config: Any
    ) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        
        # Unpack batch (assuming batch is a pytree of [B, Unroll_len+1, ...])
        # For EfficientZero, the batch structure is more complex:
        # obs_batch_ori: [B, Stack, H, W, C] (initial obs for unroll)
        # action_batch: [B, K] (actions taken during unroll)
        # target_value_prefixes: [B, K] (rewards r_k)
        # target_values: [B, K+1] (search values V(s_k))
        # target_policies: [B, K+1, Num_Actions] (search policies Pi(s_k))
        # For simplicity, let's assume `batch` is a Transition object where
        # `observation` is the initial stack, and `action` is the first action.
        # The targets `policy_target` and `value_target` are for the initial state s0.
        # The actual unrolling and loss calculation for sequence models is more involved.
        # The original EZ `update_weights` unrolls the model step-by-step.

        # We need to get the sequence of actions from the batch if it's stored that way,
        # or assume the batch provides initial_obs and action_sequence.
        # For now, let's assume `batch` is structured to provide what EfficientZeroNet.__call__ expects.
        # This part needs careful alignment with how replay_buffer stores and samples data.
        # Let's assume `batch.observation` is initial_obs_stack [B, H, W, C*S]
        # and `batch.action_sequence` is [B, K_unroll]
        # and targets are available for each step of unroll.
        
        # This is a simplified loss for one step, needs to be adapted for unrolling
        # The model's __call__ method should handle the unrolling internally.
        
        dropout_rng, new_rng_key = jax.random.split(rng_key)

        # Online model forward pass (with unrolling)
        # The model's __call__ should return initial predictions and unrolled sequence predictions
        initial_preds, unrolled_preds = self.model.apply(
            {'params': params, 'batch_stats': batch_stats},
            batch.observation, # Initial observation stack [B, H, W, C_stack]
            batch.action_sequence, # Action sequence [B, K_unroll]
            initial_lstm_state=None, # TODO: Initialize LSTM state if use_reward_lstm
            train=True,
            rngs={'dropout': dropout_rng},
            mutable=['batch_stats'] # To update BatchNorm stats
        )
        # `new_model_state` contains updated `batch_stats`
        new_model_state = unrolled_preds.pop('batch_stats', None) # If mutable was used
        if new_model_state is None: # If no batch_stats (e.g. no BatchNorm)
            new_model_state = batch_stats


        # Loss calculation (conceptual, needs to match original EZ)
        total_loss = 0.0
        metrics = {}

        # Initial step loss (s0)
        val_loss_0 = utils.categorical_cross_entropy_loss(
            initial_preds['value_logits'], batch.value_target_initial, weights=is_weights
        ) * config.train.value_loss_coeff
        pol_loss_0 = utils.categorical_cross_entropy_loss(
            initial_preds['policy_logits'], batch.policy_target_initial, weights=is_weights
        ) * config.train.policy_loss_coeff
        total_loss += val_loss_0 + pol_loss_0
        metrics['value_loss_initial'] = val_loss_0
        metrics['policy_loss_initial'] = pol_loss_0

        # Unrolled steps losses (s1 to sK)
        # `unrolled_preds` is a pytree with leading dim K_unroll
        # `batch` targets also need to be shaped [B, K_unroll, ...]
        
        # Value loss for V(h_{k+1}) vs target_value_{k+1}
        value_loss_unroll = utils.categorical_cross_entropy_loss(
            unrolled_preds['value_logits'], batch.value_targets_unroll, weights=is_weights[:, None] # Broadcast IS weights
        ) * config.train.value_loss_coeff
        
        # Policy loss for P(a|h_{k+1}) vs target_policy_{k+1}
        policy_loss_unroll = utils.categorical_cross_entropy_loss(
            unrolled_preds['policy_logits'], batch.policy_targets_unroll, weights=is_weights[:, None]
        ) * config.train.policy_loss_coeff
        
        # Reward loss for r_k vs target_reward_k
        reward_loss_unroll = utils.categorical_cross_entropy_loss(
            unrolled_preds['reward_logits'], batch.reward_targets_unroll, weights=is_weights[:, None]
        ) * config.train.reward_loss_coeff
        
        total_loss += jnp.mean(value_loss_unroll + policy_loss_unroll + reward_loss_unroll) # Mean over unroll steps
        metrics['value_loss_unroll'] = jnp.mean(value_loss_unroll)
        metrics['policy_loss_unroll'] = jnp.mean(policy_loss_unroll)
        metrics['reward_loss_unroll'] = jnp.mean(reward_loss_unroll)

        # Consistency loss (if used)
        if config.model.use_consistency_projection:
            # Online network: project current state, then predict
            # Target network: project next state (from reanalyzed observation)
            # This requires having the next observation stacks in the batch.
            # batch.next_observation_stacks_unroll [B, K, H, W, C_stack]
            
            # Embedding h_k from online model (already in unrolled_preds['embedding'])
            # Embedding h_{k+1} from online model (in unrolled_preds['next_embedding'])
            
            # Project h_{k+1} from online model (dynamics prediction)
            # This projection uses the online projection_net + predictor_net
            pred_dynamic_proj = self.model.apply(
                {'params': params, 'batch_stats': batch_stats}, # Use online params
                unrolled_preds['next_embedding'], # h_{k+1} from dynamics
                method=networks.EfficientZeroNet.project,
                train=True, with_grad=True, # Project and predict
                rngs={'dropout': dropout_rng} # Pass dropout key if predictor has dropout
            ) # Output shape [B, K, D_proj_output]

            # Project target_h_{k+1} (from re-encoding actual next observation)
            # This requires re-encoding the *actual* next observations from the buffer.
            # Let's assume `batch.next_observation_stacks_unroll` exists.
            # Target projection uses target_params and no predictor gradient.
            target_next_embeddings_reencoded = self.model.apply(
                {'params': target_params, 'batch_stats': batch_stats}, # Use target_params for representation
                batch.next_observation_stacks_unroll, # Actual o_{t+k+1} stacks
                method=networks.EfficientZeroNet.initial_inference, # Just representation part
                train=False # Target net is in eval mode
            )[0] # Get only the embedding part

            target_proj = self.model.apply(
                {'params': target_params, 'batch_stats': batch_stats}, # Use target_params for projection
                target_next_embeddings_reencoded,
                method=networks.EfficientZeroNet.project,
                train=False, with_grad=False # Project only, no predictor, no grad
            )
            target_proj = jax.lax.stop_gradient(target_proj)

            consistency_loss_val = utils.cosine_similarity_loss_jax(
                pred_dynamic_proj, target_proj, weights=is_weights[:, None] # Broadcast IS weights
            ) * config.train.consistency_loss_coeff
            total_loss += jnp.mean(consistency_loss_val)
            metrics['consistency_loss'] = jnp.mean(consistency_loss_val)

        metrics['total_loss'] = total_loss
        # Return new_model_state (containing updated batch_stats) as part of aux
        return total_loss, (metrics, new_model_state, new_rng_key)


    def _train_step_device(
        self, train_state: TrainState, 
        per_device_batch: Transition, 
        per_device_is_weights: chex.Array,
        config: Any # Static arg
    ) -> Tuple[TrainState, Dict[str, chex.Array]]:
        
        step_rng_key, loss_rng_key = jax.random.split(train_state.rng_key)

        grad_fn = jax.value_and_grad(self._loss_fn, argnums=0, has_aux=True)
        (loss, (metrics, new_batch_stats, _)), grads = grad_fn(
            train_state.params,
            train_state.target_params,
            train_state.batch_stats,
            per_device_batch,
            per_device_is_weights,
            loss_rng_key,
            config
        )
        
        # Aggregate gradients and metrics across devices
        grads = jax.lax.pmean(grads, axis_name='devices')
        metrics = jax.lax.pmean(metrics, axis_name='devices')
        # new_batch_stats are also averaged if they changed.
        # For BatchNorm, stats are updated based on local batch, then averaged.
        # This is a common way to handle BatchNorm in distributed training.
        new_batch_stats = jax.lax.pmean(new_batch_stats, axis_name='devices')

        # Apply updates
        updates, new_opt_state = self.optimizer.update(grads, train_state.opt_state, train_state.params)
        new_params = optax.apply_updates(train_state.params, updates)
        
        new_train_state = train_state.replace(
            step=train_state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            batch_stats=new_batch_stats, # Update batch_stats
            rng_key=step_rng_key # Update RNG key
        )
        metrics['loss'] = loss # Add scalar loss to metrics
        return new_train_state, metrics

    def get_pmapped_train_step(self) -> Callable:
        # Pass config as a static argument to pmap
        return jax.pmap(self._train_step_device, axis_name='devices', static_broadcasted_argnums=(3,))

    def update_target_network(self, train_state: TrainState) -> TrainState:
        # Standard EMA update or hard update
        # Hard update for simplicity here
        return train_state.replace(target_params=train_state.params)

    def update_self_play_params(self, train_state: TrainState) -> TrainState:
        return train_state.replace(self_play_params=train_state.params)

    # Self-play logic will be primarily in main.py, calling MCTS and environment.
    # This agent class focuses on the learning update.