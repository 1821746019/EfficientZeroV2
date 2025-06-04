import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import cv2
import functools

def pmap_decorator(axis_name: str = 'batch'):
    """装饰器：自动在所有可用TPU设备上并行化函数"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 自动获取设备数量并并行化
            return jax.pmap(func, axis_name=axis_name)(*args, **kwargs)
        return wrapper
    return decorator

def get_num_devices() -> int:
    """获取可用设备数量"""
    return jax.device_count()

def replicate_across_devices(pytree: Any) -> Any:
    """在所有设备上复制数据"""
    return jax.tree_map(lambda x: jnp.array([x] * get_num_devices()), pytree)

def unreplicate_from_devices(pytree: Any) -> Any:
    """从设备上取回数据（取第一个设备的数据）"""
    return jax.tree_map(lambda x: x[0], pytree)

def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """符号对数变换"""
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)

def symexp(x: jnp.ndarray) -> jnp.ndarray:
    """符号指数变换"""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

class DiscreteSupport:
    """离散支持变换，用于值函数和奖励的编码"""
    
    def __init__(self, vmin: float, vmax: float, num_atoms: int):
        self.vmin = vmin
        self.vmax = vmax
        self.num_atoms = num_atoms
        self.delta = (vmax - vmin) / (num_atoms - 1)
        self.support = jnp.linspace(vmin, vmax, num_atoms)
    
    def scalar_to_vector(self, x: jnp.ndarray) -> jnp.ndarray:
        """标量转为概率向量"""
        x = jnp.clip(x, self.vmin, self.vmax)
        b = (x - self.vmin) / self.delta
        l = jnp.floor(b).astype(jnp.int32)
        u = jnp.ceil(b).astype(jnp.int32)
        
        # 处理边界情况
        l = jnp.clip(l, 0, self.num_atoms - 1)
        u = jnp.clip(u, 0, self.num_atoms - 1)
        
        # 计算概率
        p_l = u - b
        p_u = b - l
        
        # 创建概率向量
        batch_size = x.shape[0]
        prob = jnp.zeros((batch_size, self.num_atoms))
        prob = prob.at[jnp.arange(batch_size), l].add(p_l)
        prob = prob.at[jnp.arange(batch_size), u].add(p_u)
        
        return prob
    
    def vector_to_scalar(self, logits: jnp.ndarray) -> jnp.ndarray:
        """概率向量转为标量期望值"""
        probs = jax.nn.softmax(logits, axis=-1)
        return jnp.sum(probs * self.support, axis=-1, keepdims=True)