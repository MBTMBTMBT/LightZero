from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map
import wandb

env_id = 'MsPacmanNoFrameskip-v4'
action_space_size = atari_env_action_space_map[env_id]

# ---- aligned knobs ----
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 32
batch_size = 256
max_env_step = int(2e5)
reanalyze_ratio = 0.1
chance_space_size = 8
replay_ratio = 0.25
update_per_collect = None
# --------------------------------------

atari_stochastic_muzero_config = dict(
    exp_name=f'stochastic_mz/MsPacman_v4_smz_ns{num_simulations}_rr{replay_ratio}_re{reanalyze_ratio}_chance{chance_space_size}_seed0',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=(4, 96, 96),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False),
        frame_stack_num=4,
        gray_scale=True,
    ),
    policy=dict(
        model=dict(
            observation_shape=(4, 96, 96),
            frame_stack_num=4,
            action_space_size=action_space_size,
            chance_space_size=chance_space_size,
            image_channel=1,
            gray_scale=True,
            downsample=True,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        model_path=None,
        cuda=True,
        gumbel_algo=False,
        mcts_ctree=True,
        env_type='not_board_games',
        game_segment_length=400,
        use_augmentation=True,
        update_per_collect=update_per_collect,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=1e-4,
        target_update_freq=100,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        use_wandb=False,  # to avoid calling wandb from inside
    ),
)
main_config = EasyDict(atari_stochastic_muzero_config)

create_config = EasyDict(dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='stochastic_muzero',
        import_names=['lzero.policy.stochastic_muzero'],
    ),
))

if __name__ == "__main__":
    from lzero.entry import train_muzero

    run = wandb.init(
        project=env_id,
        name=getattr(main_config, "exp_name", None),
        config=main_config,
        sync_tensorboard=True,
        settings=wandb.Settings(console="redirect"),
        monitor_gym=False,
    )

    train_muzero([main_config, create_config], seed=0, model_path=main_config.policy.get('model_path'), max_env_step=max_env_step)

    wandb.finish()
