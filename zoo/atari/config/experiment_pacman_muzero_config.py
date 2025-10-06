from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map
import wandb

_OVERRIDE_PROJECT = "MsPacmanNoFrameskip-v4"
_USE_CFG_EXP_NAME = True

_real_init = wandb.init
def _patched_init(*args, **kwargs):
    if _OVERRIDE_PROJECT:
        kwargs["project"] = _OVERRIDE_PROJECT
    if "name" not in kwargs or kwargs["name"] is None:
        run_name = None
        if _USE_CFG_EXP_NAME and "config" in kwargs and kwargs["config"] is not None:
            cfg = kwargs["config"]
            run_name = getattr(cfg, "exp_name", None) or (cfg.get("exp_name") if hasattr(cfg, "get") else None)
        kwargs["name"] = run_name
    return _real_init(*args, **kwargs)
wandb.init = _patched_init

env_id = 'MsPacmanNoFrameskip-v4'
action_space_size = atari_env_action_space_map[env_id]

# ---- aligned knobs (mirror stochastic) ----
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 32
batch_size = 256
max_env_step = int(2e5)
reanalyze_ratio = 0.1
replay_ratio = 0.25
update_per_collect = None
# -------------------------------------------

atari_muzero_config = dict(
    exp_name=f'muzero/MsPacman_v4_muzero_ns{num_simulations}_rr{replay_ratio}_re{reanalyze_ratio}_seed0',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=(4, 84, 84),
        frame_stack_num=4,
        gray_scale=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False),
    ),
    policy=dict(
        analysis_sim_norm=False,
        cal_dormant_ratio=False,
        model=dict(
            observation_shape=(4, 84, 84),
            frame_stack_num=4,
            image_channel=1,
            gray_scale=True,
            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            use_sim_norm=True,
            model_type='conv'
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400,
        random_collect_episode_num=0,
        use_augmentation=True,
        use_priority=False,
        update_per_collect=update_per_collect,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type='Adam',
        piecewise_decay_lr_scheduler=False,
        learning_rate=3e-3,
        target_update_freq=100,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        use_wandb=True,
    ),
)
main_config = EasyDict(atari_muzero_config)

create_config = EasyDict(dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
))

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
