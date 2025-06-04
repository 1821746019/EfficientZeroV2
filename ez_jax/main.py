# ==============================================================================
# ez_jax/main.py
# ==============================================================================
from ez_jax.training.trainer import EfficientZeroTrainer
from ez_jax.config.config import get_atari_config,get_dmc_config
from ez_jax.agent.ez_agent import EfficientZeroAgent

# ==============================================================================
# ez_jax/main.py - 主程序
# ==============================================================================
import argparse
import os

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EfficientZero JAX Training')
    parser.add_argument('--env', choices=['atari', 'dmc'], default='atari',
                       help='Environment type')
    parser.add_argument('--steps', type=int, default=100000,
                       help='Total training steps')
    parser.add_argument('--load', type=str, default=None,
                       help='Load model from file')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation')
    
    args = parser.parse_args()
    
    # 设置JAX环境
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['JAX_PLATFORM_NAME'] = 'tpu'
    
    # 获取配置
    if args.env == 'atari':
        config = get_atari_config()
    elif args.env == 'dmc':
        config = get_dmc_config()
    else:
        raise ValueError(f"Unknown environment: {args.env}")
    
    # 创建智能体
    agent = EfficientZeroAgent(config)
    
    # 加载模型（如果指定）
    if args.load:
        agent.load_model(args.load)
    
    if args.eval_only:
        # 仅评估
        agent.evaluate(num_episodes=100)
    else:
        # 训练
        agent.train(total_steps=args.steps)
        
        # 最终评估
        agent.evaluate(num_episodes=100)
        
        # 保存最终模型
        agent.save_model('final_model.pkl')
    
    # 清理
    agent.env.close()

if __name__ == '__main__':
    main()

