import flax.linen as nn
import jax
import jax.numpy as jnp
import chex
from typing import Any, Sequence, Tuple, Optional, Callable
from . import utils # Assuming utils.py is in the same directory

Array = chex.Array
PRNGKey = chex.PRNGKey
Shape = Tuple[int, ...]
Dtype = Any

class ResidualBlock(nn.Module):
    features: int
    norm: Callable = nn.BatchNorm # Or nn.LayerNorm

    @nn.compact
    def __call__(self, x: Array, train: bool) -> Array:
        residual = x
        y = self.norm(use_running_average=not train, name="norm1")(x)
        y = nn.relu(y)
        y = nn.Conv(features=self.features, kernel_size=(3, 3), strides=(1, 1), padding="SAME", name="conv1")(y)
        
        y = self.norm(use_running_average=not train, name="norm2")(y)
        y = nn.relu(y)
        y = nn.Conv(features=self.features, kernel_size=(3, 3), strides=(1, 1), padding="SAME", name="conv2")(y)
        
        if residual.shape != y.shape: # Handle projection shortcut if dimensions change
            residual = nn.Conv(features=self.features, kernel_size=(1, 1), strides=(1, 1), padding="SAME", name="shortcut_conv")(residual)
            residual = self.norm(use_running_average=not train, name="shortcut_norm")(residual)

        return residual + y

class DownSampler(nn.Module):
    channels: int # Output channels after first main conv block

    @nn.compact
    def __call__(self, x: Array, train: bool) -> Array:
        # Initial Conv
        x = nn.Conv(features=self.channels // 2, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="conv_ds_1")(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn_ds_1")(x)
        x = nn.relu(x)
        x = ResidualBlock(features=self.channels // 2, name="res_ds_1")(x, train)
        
        # Second Conv block
        shortcut = nn.Conv(features=self.channels, kernel_size=(1,1), strides=(2,2), padding="SAME", name="conv_ds_2_shortcut")(x)
        shortcut = nn.BatchNorm(use_running_average=not train, name="bn_ds_2_shortcut")(shortcut)

        x = nn.Conv(features=self.channels, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="conv_ds_2")(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn_ds_2")(x)
        x = nn.relu(x)
        x = ResidualBlock(features=self.channels, name="res_ds_2")(x, train)
        x = x + shortcut # Apply shortcut after residual block

        # Pooling and more resblocks
        x = nn.avg_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(features=self.channels, name="res_ds_3")(x, train)
        x = nn.avg_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        return x

class RepresentationNet(nn.Module):
    config: Any # Hydra config for model parameters

    @nn.compact
    def __call__(self, x: Array, train: bool) -> Array:
        # x shape: [B, H, W, C_stack] if channels_last, or [B, C_stack, H, W] if channels_first
        # Assuming channels_last for Flax Conv default
        
        if self.config.model.down_sample:
            x = DownSampler(channels=self.config.model.repr_cnn_channels[0], name="DownSampler")(x, train)
            # After downsampling, x has `self.config.model.repr_cnn_channels[0]` channels
            # The original EZ code has a sequence of resblocks after downsampling or initial conv
            # Let's assume the first channel dim in repr_cnn_channels is the target for these blocks
            num_res_blocks = self.config.model.num_blocks # num_blocks in original config
            current_channels = self.config.model.repr_cnn_channels[0]
        else:
            # Initial convolution if not downsampling
            current_channels = self.config.model.repr_cnn_channels[0] # Or a specific initial channel count
            x = nn.Conv(features=current_channels, kernel_size=(3,3), padding="SAME", name="initial_conv")(x)
            x = nn.BatchNorm(use_running_average=not train, name="initial_bn")(x)
            x = nn.relu(x)
            num_res_blocks = self.config.model.num_blocks

        for i in range(num_res_blocks):
            x = ResidualBlock(features=current_channels, name=f"repr_resblock_{i}")(x, train)
        
        # Flatten and project to embedding_dim
        x = x.reshape((x.shape[0], -1)) # Flatten
        x = nn.Dense(features=self.config.model.embedding_dim, name="repr_dense_out")(x)
        # Original EZ might have state_norm here. Can be nn.LayerNorm or just rely on BatchNorm.
        if self.config.model.state_norm:
             x = nn.LayerNorm(name="repr_output_layernorm")(x)
        return x

class DynamicsNet(nn.Module):
    config: Any

    @nn.compact
    def __call__(self, embedding: Array, action: Array, lstm_state: Optional[Tuple[Array, Array]], train: bool) -> Tuple[Array, Array, Optional[Tuple[Array, Array]]]:
        # embedding: [B, D_embed]
        # action: [B] (discrete) or [B, D_action] (continuous)
        
        if self.config.model.use_action_embedding:
            if action.ndim == 1: # Discrete actions
                action_embedded = nn.Embed(num_embeddings=self.config.env.action_space_size, 
                                           features=self.config.model.action_embedding_dim,
                                           name="action_embed")(action)
            else: # Continuous actions already embedded or to be passed through a Dense
                action_embedded = nn.Dense(features=self.config.model.action_embedding_dim, 
                                           name="action_embed_dense")(action)
        else: # One-hot encode discrete actions or use continuous actions directly
            if action.ndim == 1:
                action_embedded = jax.nn.one_hot(action, num_classes=self.config.env.action_space_size)
            else:
                action_embedded = action

        x = jnp.concatenate([embedding, action_embedded], axis=-1)
        
        for i, layer_size in enumerate(self.config.model.dynamics_mlp_layers):
            x = nn.Dense(features=layer_size, name=f"dynamics_dense_{i}")(x)
            x = nn.relu(x)
            # Consider adding LayerNorm here if not using BatchNorm extensively in ResBlocks
            # x = nn.LayerNorm(name=f"dynamics_ln_{i}")(x)

        next_embedding = nn.Dense(features=self.config.model.embedding_dim, name="dynamics_next_embedding_dense")(x)
        if self.config.model.state_norm: # Normalize next_embedding
            next_embedding = nn.LayerNorm(name="dynamics_next_embedding_ln")(next_embedding)

        # Reward prediction
        reward_features = x # Use features before projecting to next_embedding for reward
        
        new_lstm_state = None
        if self.config.model.use_reward_lstm:
            chex.assert_msg(lstm_state is not None, "LSTM state must be provided if use_reward_lstm is True")
            # Flax LSTMCell expects input [B, F] and returns output [B, H], new_carry (h,c)
            lstm_cell = nn.LSTMCell(features=self.config.model.reward_lstm_hidden_size, name="reward_lstm_cell")
            reward_features_for_lstm, new_lstm_state = lstm_cell(lstm_state, reward_features)
            reward_head_input = reward_features_for_lstm
        else:
            reward_head_input = reward_features

        if self.config.train.use_categorical_reward:
            num_reward_outputs = self.config.train.reward_support_bins
        else:
            num_reward_outputs = 1
        
        reward_logits = nn.Dense(features=num_reward_outputs, name="reward_head_dense")(reward_head_input)
        
        return next_embedding, reward_logits, new_lstm_state


class PredictionNet(nn.Module):
    config: Any

    @nn.compact
    def __call__(self, embedding: Array, train: bool) -> Tuple[Array, Array]:
        # embedding: [B, D_embed]
        x = embedding
        for i, layer_size in enumerate(self.config.model.prediction_mlp_layers):
            x = nn.Dense(features=layer_size, name=f"prediction_dense_{i}")(x)
            x = nn.relu(x)
            # x = nn.LayerNorm(name=f"prediction_ln_{i}")(x)

        # Value head
        if self.config.train.use_categorical_value:
            num_value_outputs = self.config.train.value_support_bins
        else:
            num_value_outputs = 1
        value_logits = nn.Dense(features=num_value_outputs, name="value_head_dense")(x)

        # Policy head
        policy_logits = nn.Dense(features=self.config.env.action_space_size, name="policy_head_dense")(x)
        
        return value_logits, policy_logits

class ProjectionNet(nn.Module):
    config: Any

    @nn.compact
    def __call__(self, embedding: Array, train: bool) -> Array:
        # embedding: [B, D_embed] (output of representation or dynamics)
        x = nn.Dense(features=self.config.model.projection_hidden_dim, name="proj_dense_1")(embedding)
        x = nn.LayerNorm(name="proj_ln_1")(x) # Original EZ uses BatchNorm, LayerNorm is often better for MLPs
        x = nn.relu(x)
        x = nn.Dense(features=self.config.model.projection_output_dim, name="proj_dense_2")(x)
        # No final activation in projection, as per BYOL/SPR common practice
        return x

class PredictorNet(nn.Module): # Renamed from ProjectionHeadNet for clarity (predicts z' from z_online)
    config: Any

    @nn.compact
    def __call__(self, projected_embedding: Array, train: bool) -> Array:
        # projected_embedding: [B, D_proj_output]
        x = nn.Dense(features=self.config.model.predictor_hidden_dim, name="pred_dense_1")(projected_embedding)
        x = nn.LayerNorm(name="pred_ln_1")(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.config.model.projection_output_dim, name="pred_dense_2")(x) # Output same dim as projection
        return x


class EfficientZeroNet(nn.Module):
    config: Any

    def setup(self):
        self.representation_net = RepresentationNet(config=self.config, name="representation")
        self.dynamics_net = DynamicsNet(config=self.config, name="dynamics")
        self.prediction_net = PredictionNet(config=self.config, name="prediction")
        
        if self.config.model.use_consistency_projection:
            self.projection_net = ProjectionNet(config=self.config, name="projection")
            self.predictor_net = PredictorNet(config=self.config, name="predictor")

    def initial_inference(self, observations: Array, train: bool) -> Tuple[Array, Array, Array]:
        """Initial inference from observations."""
        # observations: [B, H, W, C_stack] or [B, C_stack, H, W]
        embedding = self.representation_net(observations, train=train)
        value_logits, policy_logits = self.prediction_net(embedding, train=train)
        return embedding, value_logits, policy_logits

    def recurrent_inference(
        self, embedding: Array, action: Array, lstm_state: Optional[Tuple[Array, Array]], train: bool
    ) -> Tuple[Array, Array, Array, Array, Optional[Tuple[Array, Array]]]:
        """Recurrent inference for dynamics and subsequent prediction."""
        # embedding: [B, D_embed], action: [B] or [B, D_action]
        next_embedding, reward_logits, new_lstm_state = self.dynamics_net(embedding, action, lstm_state, train=train)
        next_value_logits, next_policy_logits = self.prediction_net(next_embedding, train=train)
        return next_embedding, reward_logits, next_value_logits, next_policy_logits, new_lstm_state

    def project(self, embedding: Array, train: bool, with_grad: bool = True) -> Array:
        """Projects embedding for consistency loss. `with_grad` is for the predictor."""
        if not self.config.model.use_consistency_projection:
            # Return a dummy value or raise error if not configured
            return jnp.zeros_like(embedding) # Or handle appropriately

        projected = self.projection_net(embedding, train=train)
        if with_grad: # This branch is for the "online" network's predictor
            return self.predictor_net(projected, train=train)
        else: # This branch is for the "target" network's projection (no grad through predictor)
            return projected


    def __call__(self, observations: Array, actions_sequence: Optional[Array] = None, initial_lstm_state: Optional[Tuple[Array,Array]] = None, train: bool = False):
        """
        Main call method for training, unrolling the model.
        observations: [B, H, W, C_stack] (initial observations)
        actions_sequence: [B, K, D_action] (sequence of K actions to unroll)
                          If None, only initial_inference is performed.
        initial_lstm_state: Tuple for LSTM if use_reward_lstm is True.
        train: bool, for BatchNorm and Dropout
        """
        embedding, value_logits, policy_logits = self.initial_inference(observations, train=train)

        if actions_sequence is None:
            # This case is for when only initial_inference is needed (e.g. by mctx root_fn)
            # For reward_lstm, we need to decide what to return.
            # If dynamics_net is stateful (has LSTM), then its initial state needs to be handled.
            # For now, assume initial_lstm_state is for the reward LSTM if used.
            
            # We need to return reward_logits for the initial state as well for mctx.
            # This implies the prediction_net might also need to predict initial reward, or
            # we run a dummy dynamics step.
            # For simplicity, let's assume initial_inference provides what mctx root_fn needs.
            # MCTX RootFnOutput: prior_logits, value, embedding
            # The 'value' from initial_inference is V(s_0).
            # The 'embedding' is h_0.
            # 'prior_logits' is P(a|s_0).
            return embedding, value_logits, policy_logits # No reward or next_lstm_state here

        # Unrolling for training
        chex.assert_rank(actions_sequence, 2) # [B, K] or [B, K, D_action_embed]
        num_unroll_steps = actions_sequence.shape[1]
        
        outputs = []
        current_embedding = embedding
        current_lstm_state = initial_lstm_state

        for k in range(num_unroll_steps):
            action_k = actions_sequence[:, k]
            (next_embedding, 
             reward_logits_k, 
             next_value_logits_k, 
             next_policy_logits_k, 
             next_lstm_state_k) = self.recurrent_inference(current_embedding, action_k, current_lstm_state, train=train)
            
            # For consistency loss target
            projected_target_embedding_k = None
            if self.config.model.use_consistency_projection:
                # This projection should be on the *next_embedding* from dynamics, using target network weights
                # So, this __call__ is for the online network. Target projection happens in loss_fn.
                pass # Placeholder, consistency handled in loss_fn

            outputs.append({
                "embedding": current_embedding, # h_k
                "next_embedding": next_embedding, # h_{k+1} from dynamics(h_k, a_k)
                "reward_logits": reward_logits_k, # r_k
                "value_logits": next_value_logits_k, # V(h_{k+1})
                "policy_logits": next_policy_logits_k, # P(a|h_{k+1})
            })
            current_embedding = next_embedding
            current_lstm_state = next_lstm_state_k
            
        # Collate outputs: stack arrays from the list of dicts
        collated_outputs = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=1), *outputs)
        # Also return initial predictions
        initial_predictions = {
            "embedding": embedding,
            "value_logits": value_logits,
            "policy_logits": policy_logits
        }
        return initial_predictions, collated_outputs