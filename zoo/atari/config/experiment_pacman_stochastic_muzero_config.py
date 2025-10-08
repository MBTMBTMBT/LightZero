# experiment_pacman_stochastic_muzero_config.py
import re

from easydict import EasyDict
import argparse, os
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
parser.add_argument("--out_dir", type=str, default=".", help="root dir to place exp outputs")
args = parser.parse_args()

env_id = "MsPacmanNoFrameskip-v4"

try:
    from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map
    action_space_size = atari_env_action_space_map.get(env_id)
except Exception:
    action_space_size = None

if action_space_size is None:
    import gymnasium as gym
    env = gym.make(env_id, render_mode="rgb_array")
    try:
        action_space_size = env.action_space.n
    finally:
        env.close()

collector_env_num = 16
n_episode = 16
evaluator_env_num = 10
n_evaluator_episode = 20

num_simulations = 50
batch_size = 256
max_env_step = 400_000
replay_ratio = 0.25
target_update_freq = 100
game_segment_length = 400
replay_buffer_size = int(1e6)
reanalyze_ratio = 0.0
update_per_collect = None

# stochastic-specific
chance_space_size = 8

obs_shape = (4, 96, 96)

# --- wandb: external init (do not let LightZero init wandb) ---
project_name = env_id
exp_tag = f"stmz/pacman_seed{args.seed}"
exp_name = os.path.join(args.out_dir, exp_tag)

main_cfg = EasyDict(dict(
    exp_name=exp_name,
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=obs_shape,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=n_evaluator_episode,
        manager=dict(shared_memory=False),
        frame_stack_num=4,
        gray_scale=True,
        manually_discretization=False,
        discrete_action_list=None,
        continuous=False,
    ),
    policy=dict(
        model=dict(
            observation_shape=obs_shape,
            frame_stack_num=4,
            action_space_size=action_space_size,
            chance_space_size=chance_space_size,
            image_channel=1,
            gray_scale=True,
            downsample=True,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type="one_hot",
            norm_type="BN",
        ),
        model_path=None,
        cuda=True,
        gumbel_algo=False,
        mcts_ctree=True,
        env_type="not_board_games",
        game_segment_length=game_segment_length,
        use_augmentation=True,
        update_per_collect=update_per_collect,
        replay_ratio=replay_ratio,
        batch_size=batch_size,
        optim_type = "Adam",
        piecewise_decay_lr_scheduler = False,
        learning_rate = 1e-4,
        target_update_freq=target_update_freq,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=replay_buffer_size,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        use_wandb=False,  # do not init wandb inside LightZero
    ),
))

create_cfg = EasyDict(dict(
    env=dict(
        type="atari_lightzero",
        import_names=["zoo.atari.envs.atari_lightzero_env"],
    ),
    env_manager=dict(type="subprocess"),
    policy=dict(type="stochastic_muzero", import_names=["lzero.policy.stochastic_muzero"]),
))


def _wb_sanitize(s: str, maxlen: int = 128, fallback: str = "exp"):
    """
    Make a string safe for W&B project/run names.
    - Replace disallowed chars [/ \ # ? % :] with '-'
    - Keep only [A-Za-z0-9 _ - .] afterwards
    - Collapse repeats, trim leading/trailing separators
    - Ensure it starts with alnum; fallback if empty
    """
    if s is None:
        return fallback
    s = s.replace(" ", "_")
    s = re.sub(r"[\/\\#\?\%:]+", "-", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "-", s)
    s = re.sub(r"[-_\.]{2,}", "-", s)
    s = s.strip("-_.")
    if not s or not re.match(r"[A-Za-z0-9]", s[0]):
        s = f"{fallback}-{s}" if s else fallback
    return s[:maxlen]


if __name__ == "__main__":
    from lzero.entry import train_muzero
    project_name = _wb_sanitize(env_id, fallback="atari")
    run = wandb.init(
        project=project_name,
        name=os.path.basename(exp_name),
        config=main_cfg,
        sync_tensorboard=True,
        settings=wandb.Settings(console="redirect"),
        monitor_gym=False,
    )
    train_muzero([main_cfg, create_cfg], seed=args.seed, max_env_step=max_env_step)
    wandb.finish()
