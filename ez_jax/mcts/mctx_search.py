import mctx
from typing import NamedTuple, Callable
import jax.numpy as jnp

class MCTSOutput(NamedTuple):
    """MCTS搜索输出"""
    search_values: jnp.ndarray
    search_policies: jnp.ndarray
    best_actions: jnp.ndarray

class GumbelMuZeroFns:
    """Gumbel MuZero的函数集合，用于mctx"""
    
    def __init__(self, model, value_support, reward_support, num_simulations: int = 16):
        self.model = model
        self.value_support = value_support
        self.reward_support = reward_support
        self.num_simulations = num_simulations
    
    def root_fn(self, root_state, value, policy_logits):
        """根节点函数"""
        return mctx.RootFnOutput(
            prior_logits=policy_logits,
            value=value,
            embedding=root_state
        )
    
    def recurrent_fn(self, params, rng_key, action, embedding):
        """递归函数"""
        next_state, reward_logits, value_logits, policy_logits = self.model.apply(
            params, embedding, action, method=self.model.recurrent_inference
        )
        
        # 转换支持表示到标量
        reward = self.reward_support.vector_to_scalar(reward_logits)
        value = self.value_support.vector_to_scalar(value_logits)
        
        return mctx.RecurrentFnOutput(
            reward=reward.squeeze(-1),
            discount=jnp.ones_like(reward.squeeze(-1)) * 0.997,  # discount factor
            prior_logits=policy_logits,
            value=value.squeeze(-1)
        ), next_state
    
    def search(self, params, rng_key, root_states, root_values, root_policies):
        """执行MCTS搜索"""
        batch_size = root_states.shape[0]
        
        # 创建根节点
        root = self.root_fn(root_states, root_values, root_policies)
        
        # 执行Gumbel搜索
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=self.recurrent_fn,
            num_simulations=self.num_simulations,
            max_depth=None,
            gumbel_scale=1.0
        )
        
        # 提取搜索结果
        search_values = policy_output.search_tree.summary().value
        search_policies = policy_output.action_weights
        best_actions = policy_output.action
        
        return MCTSOutput(
            search_values=search_values,
            search_policies=search_policies,
            best_actions=best_actions
        )

