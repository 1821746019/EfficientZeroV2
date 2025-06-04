import jax
import jax.numpy as jnp
import numpy as np
import flax.jax_utils as flax_utils
from flax.training import checkpoints
import optax
import chex
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import time
import wandb
from functools import partial
from typing import Callable, List, Tuple, Dict
from . import agent as agent_lib # Renamed to avoid conflict
from . import environments
from . import replay_buffer as rb_lib
from . import utils
from . import mcts_config as mcts_cfg # Renamed
from eval import run_evaluation
def train_step_wrapper(pmapped_train_step_fn, sharded_train_state, sharded_batch_transitions, sharded_is_weights, sharded_rng_keys):
    """Helper to unpack batch for the agent's train_step."""
    # The agent._train_step_device expects (train_state, per_device_batch, per_device_is_weights, config)
    # Here, per_device_batch is sharded_batch_transitions.
    # config is static, so it's broadcasted by pmap.
    return pmapped_train_step_fn(sharded_train_state, sharded_batch_transitions, sharded_is_weights, sharded_rng_keys)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Initialize W&B
    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name if cfg.wandb.name else f"{cfg.env.game}-jax-{time.strftime('%Y%m%d-%H%M%S')}",
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            tags=list(cfg.wandb.tags) if cfg.wandb.tags else [],
            mode=cfg.wandb.mode
        )

    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    
    # Setup PRNG & Devices
    rng_seq = utils.PRNGSequence(jax.random.PRNGKey(cfg.seed))
    devices = jax.local_devices() # Use local devices, typically all TPU cores on a pod slice
    num_devices = len(devices)
    print(f"Using {num_devices} JAX devices: {devices}")

    # Create a dummy environment to get observation and action specs
    # This env is only for spec, not for actual interaction here
    dummy_env_cfg_for_spec = OmegaConf.to_container(cfg, resolve=True) # Ensure it's a dict
    # Ensure frame_stack_axis is correctly set for spec if needed, but create_atari_env handles it.
    spec_env = environments.create_atari_env(cfg.env.game, 0, cfg)
    obs_spec = spec_env.observation_space.shape
    action_dim = spec_env.action_space.n
    spec_env.close()
    print(f"Observation spec: {obs_spec}, Action dim: {action_dim}")

    # Initialize Agent and TrainState
    agent = agent_lib.EfficientZeroAgent(cfg, obs_spec, action_dim)
    initial_train_state_key = rng_seq.next()
    train_state = agent.initial_train_state(initial_train_state_key)
    
    # Replicate TrainState across devices
    sharded_train_state = flax_utils.replicate(train_state, devices=devices)
    
    # Get the pmapped training step function
    # Pass the static config to the pmapped function
    pmapped_train_step_fn = agent.get_pmapped_train_step()
    
    # Initialize Replay Buffer (on host CPU)
    replay_buffer_state = rb_lib.init_replay_buffer(
        capacity=cfg.replay_buffer.capacity,
        obs_spec=obs_spec, # Stacked observation shape
        action_dim=action_dim,
        config=cfg
    )

    # Setup self-play environments (one set per device for parallel data collection)
    num_envs_per_device = cfg.actors.num_envs_per_device
    total_self_play_envs = num_envs_per_device * num_devices
    
    # For self-play, we'll run episodes sequentially on each device for now.
    # A more advanced setup might use jax.experimental.pmap_with_aux for env steps.
    # Or, run self-play in separate Python processes / threads if I/O bound.
    # For pure JAX, let's assume a pmapped self-play function.
    
    # Actor function for self-play (to be pmapped)
    # This is a simplified actor loop for one device managing multiple envs.
    @partial(jax.pmap, axis_name='devices', static_broadcasted_argnums=(2,3,4,5,6))
    def pmapped_self_play_actor_loop(
        actor_params_and_stats: chex.ArrayTree, # Contains {'params': ..., 'batch_stats': ...}
        actor_rng_keys: chex.PRNGKey, # Per-device RNG
        # Static args:
        config: DictConfig, 
        num_envs_on_this_device: int,
        model_apply_fn_static: Callable, # The model.apply method
        obs_spec_static: Tuple,
        action_dim_static: int
    ) -> Tuple[List[Dict], chex.PRNGKey]: # Returns list of trajectories and new RNGs

        collected_trajectories_on_device = []
        
        # Create envs for this device
        device_envs = [
            environments.create_atari_env(config.env.game, int(jax.random.randint(actor_rng_keys[i], (), 0, 2**30)), config)
            for i in range(num_envs_on_this_device)
        ]
        
        # Initial states for envs on this device
        current_observations_stacked = []
        for env in device_envs:
            obs, _ = env.reset()
            current_observations_stacked.append(obs) # obs is already stacked
        
        # Convert to JAX array for processing, assuming all envs start together
        # This part is tricky with pmap if envs have different lengths.
        # For simplicity, assume we run a fixed number of steps or one episode per call.
        # Let's aim for one episode per env per call for this example.
        
        # This loop structure is conceptual for pmap.
        # A more robust way is to have the actor function return after N steps or 1 episode.
        # For now, let's assume this function is called to generate a batch of experience.
        
        # This simplified actor only runs MCTS for one step for each env to show the idea.
        # A full actor would loop until episodes end or enough data is collected.
        
        # Batch observations for MCTS on this device
        if not current_observations_stacked: # Should not happen if num_envs_on_this_device > 0
             return collected_trajectories_on_device, actor_rng_keys

        batched_obs_device = jnp.stack(current_observations_stacked)
        batched_obs_device_processed = jnp.array(batched_obs_device, dtype=jnp.float32) / 255.0

        mcts_keys_device = jax.random.split(actor_rng_keys[0], num_envs_on_this_device) # Use first key, split for envs

        # Run MCTS for all envs on this device in a batch
        policy_outputs = mcts_cfg.run_mcts(
            params=actor_params_and_stats, # Pass params and batch_stats
            rng_key=mcts_keys_device[0], # Use one key for the MCTS batch on this device
            observation_stack=batched_obs_device_processed,
            model_apply_fn=model_apply_fn_static,
            config=config
        )
        
        actions_for_envs = np.array(policy_outputs.action) # [Num_Envs_Device]
        policy_targets_for_envs = np.array(policy_outputs.action_weights) # [Num_Envs_Device, Num_Actions]
        
        # Extract root values from MCTS tree for value targets
        # tree.node_values shape [B, N_nodes], root is index 0
        value_targets_for_envs = np.array(policy_outputs.search_tree.node_values[:, 0])


        # Step environments (sequentially on this device for now)
        # This is where true async/parallel env stepping would be beneficial if Python envs are slow.
        new_actor_rng_keys = [] # Store new RNG keys for next iteration
        for i in range(num_envs_on_this_device):
            env = device_envs[i]
            action_to_take = actions_for_envs[i]
            
            # Store pre-decision state for replay buffer
            prev_obs_stacked = current_observations_stacked[i]

            next_obs, reward, terminated, truncated, _ = env.step(action_to_take)
            done = terminated or truncated
            
            # Create a transition dictionary
            # The policy_target and value_target are for `prev_obs_stacked`
            transition_data = {
                'observation': prev_obs_stacked,
                'action': action_to_take,
                'reward': float(reward),
                'discount': 1.0 - float(done), # Discount for Bellman backup
                'next_observation': next_obs, # This will be stacked by FrameStack
                'done': bool(done),
                'policy_target': policy_targets_for_envs[i],
                'value_target': value_targets_for_envs[i] # This is V(s_t) from MCTS
            }
            collected_trajectories_on_device.append(transition_data)

            if done:
                current_observations_stacked[i], _ = env.reset()
            else:
                current_observations_stacked[i] = next_obs
            
            # Update RNG key for this env for next call (if stateful RNGs per env)
            # For now, just pass back the main device RNG key, split again later.
        
        for env in device_envs: # Close envs if they are created per call
            env.close()

        return collected_trajectories_on_device, actor_rng_keys # Return original keys for simplicity now


    # --- Training Loop ---
    print("Starting training loop...")
    total_steps_done = 0
    # Initial model sync for self-play
    sharded_train_state = sharded_train_state.replace(
        self_play_params=sharded_train_state.params,
        # batch_stats for self_play_params should also be synced if model uses them
        # For simplicity, assume self_play_params uses the same batch_stats as online model
        # Or, self_play_params are used with train=False, so batch_stats are not updated.
    )


    for global_step_idx in range(1, cfg.total_training_steps + 1):
        loop_start_time = time.time()

        # --- Self-Play Phase ---
        actor_loop_rng_keys = jax.random.split(rng_seq.next(), num_devices)
        
        # Extract params and batch_stats for actors (from one replica, they are identical)
        # Actors use self_play_params
        actor_params_host = flax_utils.unreplicate(sharded_train_state.self_play_params)
        actor_batch_stats_host = flax_utils.unreplicate(sharded_train_state.batch_stats)
        actor_params_and_stats_replicated = flax_utils.replicate(
            {'params': actor_params_host, 'batch_stats': actor_batch_stats_host}, devices
        )

        # This call collects one "batch" of transitions from all pmapped actors
        # Each device actor returns a list of transitions.
        # The output `collected_transitions_sharded` will be a list (outer for devices)
        # of lists (inner for transitions from that device's envs).
        collected_transitions_sharded, _ = pmapped_self_play_actor_loop(
            actor_params_and_stats_replicated,
            actor_loop_rng_keys,
            # Static args:
            cfg, # Pass the full config
            num_envs_per_device,
            agent.model.apply, # Pass the raw apply function
            obs_spec,
            action_dim
        )
        
        # Flatten the list of lists and add to replay buffer
        all_new_transitions = []
        for device_transitions in collected_transitions_sharded: # device_transitions is already on host
            all_new_transitions.extend(device_transitions)
        
        if all_new_transitions:
            replay_buffer_state = rb_lib.add_transitions_to_buffer(replay_buffer_state, all_new_transitions)
        total_steps_done += len(all_new_transitions) # Count environment steps

        # --- Training Phase ---
        if replay_buffer_state.current_size >= cfg.train.start_train_after_steps:
            # Update beta for PER
            current_beta = rb_lib.get_beta_for_schedule(global_step_idx, cfg)
            replay_buffer_state = replay_buffer_state.replace(beta=current_beta)

            # Sample batch from replay buffer
            sample_key = rng_seq.next()
            # Global batch size for sampling
            global_batch_size = cfg.train.batch_size_per_device * num_devices
            
            sampled_indices_host, batch_transitions_host, is_weights_host = rb_lib.sample_batch_from_buffer(
                replay_buffer_state, sample_key, global_batch_size
            )
            
            # Shard data for devices
            # `shard_pytree` splits the leading axis.
            sharded_batch_transitions = utils.shard_pytree(batch_transitions_host, devices)
            sharded_is_weights = utils.shard_pytree(is_weights_host, devices)
            
            # Training step RNG keys (one per device)
            train_step_rng_keys = jax.random.split(rng_seq.next(), num_devices)
            
            # Execute pmapped training step
            sharded_train_state, metrics = train_step_wrapper(
                pmapped_train_step_fn,
                sharded_train_state,
                sharded_batch_transitions,
                sharded_is_weights,
                train_step_rng_keys
            )
            
            # Update priorities in replay buffer
            # Metrics might contain new_priorities if loss_fn calculates them
            # For now, assuming loss_fn doesn't directly output TD errors for priorities.
            # This part needs to be implemented based on how TD errors are computed.
            # Example: if metrics['td_error'] is returned:
            # new_priorities_host = flax_utils.unreplicate(metrics['td_error']).flatten()
            # replay_buffer_state = rb_lib.update_priorities_in_buffer(
            #     replay_buffer_state, sampled_indices_host, new_priorities_host
            # )
            pass # Placeholder for priority update

            # Log metrics (from first device, as they are pmean-ed)
            if global_step_idx % cfg.log_interval_steps == 0 and cfg.wandb.mode != "disabled":
                log_metrics = flax_utils.unreplicate(metrics)
                log_metrics_wandb = {f"train/{k}": v for k,v in log_metrics.items()}
                log_metrics_wandb["train/replay_buffer_size"] = replay_buffer_state.current_size
                log_metrics_wandb["train/total_env_steps"] = total_steps_done
                log_metrics_wandb["train/learning_rate"] = agent.optimizer.learning_rate(sharded_train_state.step[0]) # Get LR from one replica
                log_metrics_wandb["train/priority_beta"] = current_beta
                wandb.log(log_metrics_wandb, step=global_step_idx)
        else:
            print(f"Step {global_step_idx}: Filling replay buffer ({replay_buffer_state.current_size}/{cfg.train.start_train_after_steps})")


        # Update target network
        if global_step_idx % cfg.train.target_network_update_period == 0:
            sharded_train_state = jax.pmap(agent.update_target_network)(sharded_train_state)
            print(f"Step {global_step_idx}: Target network updated.")

        # Update self-play model parameters
        if global_step_idx % cfg.train.self_play_model_update_period == 0:
            sharded_train_state = jax.pmap(agent.update_self_play_params)(sharded_train_state)
            print(f"Step {global_step_idx}: Self-play model parameters updated.")

        # Checkpointing
        if global_step_idx % cfg.checkpoint_interval_steps == 0:
            unrep_train_state = flax_utils.unreplicate(sharded_train_state)
            ckpt_path = checkpoints.save_checkpoint(
                ckpt_dir=os.path.join(cfg.save_dir, "checkpoints"),
                target={'params': unrep_train_state.params, 
                        'batch_stats': unrep_train_state.batch_stats,
                        'opt_state': unrep_train_state.opt_state,
                        'step': unrep_train_state.step}, # Save relevant parts
                step=unrep_train_state.step.item(), # Use item() to get Python int
                overwrite=True,
                keep=3
            )
            print(f"Step {global_step_idx}: Checkpoint saved at {ckpt_path}")

        # Evaluation
        if global_step_idx % cfg.eval_interval_steps == 0:
            print(f"Step {global_step_idx}: Starting evaluation...")
            eval_params_host = flax_utils.unreplicate(sharded_train_state.params)
            eval_batch_stats_host = flax_utils.unreplicate(sharded_train_state.batch_stats)
            # Create a temporary path for eval model params
            temp_eval_ckpt_dir = os.path.join(cfg.save_dir, "temp_eval_ckpt")
            eval_model_path = checkpoints.save_checkpoint(
                ckpt_dir=temp_eval_ckpt_dir,
                target={'params': eval_params_host, 'batch_stats': eval_batch_stats_host},
                step=0, # Dummy step for this temp checkpoint
                overwrite=True
            )
            
            eval_rng = rng_seq.next()
            eval_metrics = run_evaluation( # Assuming run_evaluation is in utils or eval.py
                config=cfg,
                model_params_path=eval_model_path, # Pass dir, restore_checkpoint finds latest
                eval_rng_key=eval_rng,
                num_episodes=cfg.eval.num_episodes,
                video_save_dir=os.path.join(cfg.save_dir, f"videos_step_{global_step_idx}")
            )
            if cfg.wandb.mode != "disabled":
                wandb.log(eval_metrics, step=global_step_idx)
            print(f"Step {global_step_idx}: Evaluation complete. Metrics: {eval_metrics}")


        loop_duration = time.time() - loop_start_time
        print(f"Global Step: {global_step_idx}, Env Steps: {total_steps_done}, Loop Time: {loop_duration:.2f}s")

    print("Training finished.")
    if cfg.wandb.mode != "disabled":
        wandb.finish()

if __name__ == "__main__":
    main()