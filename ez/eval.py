import jax
import jax.numpy as jnp
import numpy as np
import chex
from typing import Any, Optional
from flax.training import checkpoints
import imageio
import os

from . import networks
from . import environments
from . import mcts_config as mcts_cfg
from . import utils

def run_evaluation(
    config: Any,
    model_params_path: str,
    eval_rng_key: chex.PRNGKey,
    num_episodes: int,
    video_save_dir: Optional[str] = None
):
    print(f"Starting evaluation for {num_episodes} episodes...")
    print(f"Loading model from: {model_params_path}")

    # Initialize model
    model = networks.EfficientZeroNet(config=config)
    
    # Create dummy observation and action for model initialization if needed for restore
    # This depends on how checkpoints are saved/loaded with Flax.
    # Typically, restore_checkpoint just needs the target pytree structure.
    # We can get this from an uninitialized TrainState or a dummy one.
    
    # Load model parameters
    # Assuming params are saved directly, not the full TrainState for eval
    restored_params = checkpoints.restore_checkpoint(model_params_path, target=None)
    if restored_params is None:
        raise FileNotFoundError(f"No checkpoint found at {model_params_path}")
    
    # If full TrainState was saved, extract params: restored_params = restored_state.params
    # For this example, assume only params dict is saved.
    # If batch_stats are needed for eval (e.g. BatchNorm in eval mode), they also need to be loaded.
    # Let's assume for eval, batch_stats are part of the loaded params if model uses them.
    # Or, they are loaded separately. For simplicity, we'll assume params are enough for eval mode.
    # A common practice is to save {'params': params, 'batch_stats': batch_stats}
    
    # For BatchNorm in eval, we need batch_stats.
    # If they are not part of `restored_params`, we might need to init model to get their structure.
    # Let's assume `restored_params` is a dict like {'params': ..., 'batch_stats': ...}
    # Or, if only params were saved, then batch_stats might not be used or are fixed.
    
    # For simplicity, let's assume the loaded checkpoint contains a dict with 'params'
    # and potentially 'batch_stats' if the model uses them.
    # If model.init was used to save, it would be {'params': ..., 'batch_stats': ...}
    # If only model.params were saved, then batch_stats are not available.
    
    # Let's assume the checkpoint is a dict containing 'params' and 'batch_stats'
    # If not, this part needs adjustment based on how checkpoints are saved.
    if 'params' not in restored_params: # If the raw params were saved directly
        eval_params = {'params': restored_params}
    else:
        eval_params = restored_params # Assumes dict {'params': ..., 'batch_stats': ...}

    @jax.jit
    def select_action_eval(
        params_with_stats: chex.ArrayTree, 
        observation_stack: chex.Array, 
        rng_key: chex.PRNGKey
    ) -> chex.Array:
        if config.eval.use_mcts_in_eval:
            policy_output = mcts_cfg.run_mcts(
                params=params_with_stats, # Pass params and batch_stats
                rng_key=rng_key,
                observation_stack=jnp.expand_dims(observation_stack, axis=0), # Add batch dim
                model_apply_fn=model.apply, # Pass the model's apply method
                config=config,
                num_simulations=config.eval.eval_mcts_simulations,
                temperature=config.eval.eval_temperature # For Gumbel scale in MCTS
            )
            action = policy_output.action[0] # Remove batch dim
        else: # Greedy from policy head
            # Ensure model.apply is called with batch_stats if model uses them
            _, _, policy_logits = model.apply(
                params_with_stats, # This should include {'params': ..., 'batch_stats': ...}
                jnp.expand_dims(observation_stack, axis=0),
                method=networks.EfficientZeroNet.initial_inference,
                train=False # Evaluation mode
            )
            action = jnp.argmax(policy_logits[0], axis=-1)
        return action

    episode_rewards = []
    episode_lengths = []

    for i in range(num_episodes):
        eval_rng_key, env_seed_key = jax.random.split(eval_rng_key)
        env_seed = int(jax.random.randint(env_seed_key, (), 0, 2**31 -1))

        video_file = None
        if video_save_dir:
            os.makedirs(video_save_dir, exist_ok=True)
            video_file = os.path.join(video_save_dir, f"episode_{i}_seed{env_seed}.mp4")
            
        env = environments.create_atari_env(
            game_name=config.env.game,
            seed=env_seed,
            config=config,
            record_video_path=video_file # Pass video path to wrapper
        )
        
        current_episode_reward = 0.0
        current_episode_length = 0
        
        obs, _ = env.reset() # obs is already stacked by FrameStack wrapper
        
        terminated = truncated = False
        
        frames_for_video = []

        while not (terminated or truncated):
            if video_file:
                # Gymnasium's RecordVideo wrapper handles rendering.
                # If manual rendering is needed:
                # frame = env.render() # mode="rgb_array"
                # frames_for_video.append(frame)
                pass


            eval_rng_key, action_rng_key = jax.random.split(eval_rng_key)
            
            # Normalize observation if necessary (e.g. uint8 to float32 / 255.0)
            # Assuming FrameStack gives [H, W, C*N_stack] uint8
            obs_processed = jnp.array(obs, dtype=jnp.float32) / 255.0
            
            action = select_action_eval(eval_params, obs_processed, action_rng_key)
            action_np = np.array(action) # Convert to NumPy for Gym environment

            obs, reward, terminated, truncated, _ = env.step(action_np)
            
            current_episode_reward += reward
            current_episode_length += 1
        
        episode_rewards.append(current_episode_reward)
        episode_lengths.append(current_episode_length)
        print(f"Episode {i+1}/{num_episodes}: Reward={current_episode_reward}, Length={current_episode_length}")

        env.close() # Closes video recorder if active

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print("\n--- Evaluation Summary ---")
    print(f"Average Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Average Length: {mean_length:.2f}")
    print("------------------------\n")

    return {"eval/mean_reward": mean_reward, "eval/std_reward": std_reward, "eval/mean_length": mean_length}

# Example usage (called from main.py or standalone)
if __name__ == '__main__':
    # This part would require Hydra to load config, or manual config setup
    # For a quick test, you might mock a config object
    # And ensure a checkpoint exists at the specified path
    print("Eval script can be run standalone with proper config and checkpoint.")
    # Example:
    # import hydra
    # from omegaconf import DictConfig
    #
    # @hydra.main(config_path="configs", config_name="config", version_base=None)
    # def main_eval(cfg: DictConfig):
    #     rng_key = jax.random.PRNGKey(cfg.seed)
    #     # Specify checkpoint path, e.g., from cfg or hardcoded for test
    #     checkpoint_to_eval = "path/to/your/checkpoint_dir/checkpoint_XXX" 
    #     video_dir = os.path.join(cfg.save_dir, "eval_videos")
    #     run_evaluation(cfg, checkpoint_to_eval, rng_key, cfg.eval.num_episodes, video_dir)
    #
    # main_eval()