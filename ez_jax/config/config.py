def get_atari_config():
    """Atari配置"""
    return {
        'env': {
            'name': 'ALE/BeamRider',
            'obs_shape': (84, 84, 4),
            'action_space_size': 9,
            'is_continuous': False,
            'height': 84,
            'width': 84,
            'grayscale': True,
            'frame_stack': 4
        },
        'model': {
            'num_channels': 64,
            'num_blocks': 1,
            'value_support_size': 601,
            'reward_support_size': 601,
            'value_min': -300,
            'value_max': 300,
            'reward_min': -5,
            'reward_max': 5
        },
        'mcts': {
            'num_simulations': 50,
            'num_top_actions': 16
        },
        'training': {
            'batch_size': 32,
            'unroll_steps': 5,
            'total_steps': 100000,
            'eval_interval': 1000,
            'save_interval': 5000,
            'min_trajectories': 100
        },
        'optimizer': {
            'lr': 0.0001
        },
        'buffer': {
            'max_size': 50000
        },
        'data': {
            'num_collectors': 4,
            'max_trajectory_length': 1000,
            'max_episode_steps': 27000,
            'temperature': 1.0
        },
        'eval': {
            'max_episode_steps': 27000
        },
        'seed': 42
    }

def get_dmc_config():
    """DMC配置"""
    return {
        'env': {
            'name': 'CartPole-v1',  # 简化版本，实际应该是DMC环境
            'obs_shape': (8,),
            'action_space_size': 2,
            'is_continuous': False
        },
        'model': {
            'num_channels': 64,
            'num_blocks': 2,
            'value_support_size': 601,
            'reward_support_size': 601,
            'value_min': -300,
            'value_max': 300,
            'reward_min': -10,
            'reward_max': 10
        },
        'mcts': {
            'num_simulations': 16,
            'num_top_actions': 8
        },
        'training': {
            'batch_size': 64,
            'unroll_steps': 5,
            'total_steps': 50000,
            'eval_interval': 500,
            'save_interval': 2500,
            'min_trajectories': 50
        },
        'optimizer': {
            'lr': 0.001
        },
        'buffer': {
            'max_size': 10000
        },
        'data': {
            'num_collectors': 2,
            'max_trajectory_length': 500,
            'max_episode_steps': 1000,
            'temperature': 1.0
        },
        'eval': {
            'max_episode_steps': 1000
        },
        'seed': 42
    }

