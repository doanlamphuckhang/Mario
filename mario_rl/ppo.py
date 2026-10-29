"""PPO (Proximal Policy Optimization, bản clip) viết từ đầu bằng PyTorch.

Quy trình mỗi vòng cập nhật (update):
  1. Thu thập `rollout_len` bước trên `num_envs` môi trường song song bằng chính sách hiện tại.
  2. Tính ước lượng lợi thế GAE(γ, λ) và return mục tiêu cho critic.
  3. Tối ưu hàm mục tiêu clip trên dữ liệu đó trong `epochs` lượt x `minibatches` minibatch.
"""
from __future__ import annotations

import csv
import json
import time
from contextlib import ExitStack
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .envs import SubprocVecEnv, ACTION_SETS, parse_stages
from .score_reward import ScoreReward
from .transfer import initialize_policy
from .models import ActorCritic
from .run_io import run_lock, atomic_save, restore_best_score, reconcile_logs
from .drive_backup import backup_run


@dataclass
class PPOConfig:
    # --- thí nghiệm
    run_name: str = "ppo_1-1"
    out_dir: str = "runs"
    seed: int = 1
    total_steps: int = 5_000_000      # tổng số bước agent (mỗi bước = frame_skip khung hình game)
    # --- môi trường
    stages: str = "1-1"               # "1-1" hoặc danh sách "1-1,1-2,1-3,1-4"
    actions: str = "simple"
    frame_skip: int = 4
    reward_scale: float = 0.01
    flag_bonus: float = 50.0
    death_penalty: float = 0.0
    score_weight: float = 0.0       # raw bonus / in-game score point; 0 preserves old runs
    coin_bonus: float = 0.0         # optional EXTRA coin bonus (score already includes coins)
    noop_max: int = 0
    num_envs: int = 16
    init_from: str = ""             # new experiment, feature/actor transfer; not resume
    device: str = "auto"
    # --- PPO
    rollout_len: int = 128
    epochs: int = 4
    minibatches: int = 8
    lr: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    clip_vloss: bool = True
    target_kl: float | None = None
    # --- dừng sớm: khi MỌI màn có ít nhất stop_min_episodes ván gần nhất và tỉ lệ về đích >= stop_flag_rate
    stop_flag_rate: float = 0.0       # 0 = tắt, chạy đủ total_steps
    stop_min_episodes: int = 100      # tối đa 100 (cửa sổ thống kê theo màn)
    # --- ghi log / checkpoint
    snapshot_every: int = 500_000     # lưu checkpoint mốc để xem agent tiến bộ theo thời gian
    resume: bool = False
    drive_backup_dir: str = ""       # thư mục đã mount; rỗng = không sao lưu
    backup_interval: int = 300        # giây giữa các bản sao (kiểm tra cuối update)
    backup_keep: int = 0              # số bản sao Drive mới nhất được giữ; 0 = giữ tất cả


METRIC_FIELDS = [
    "update", "global_step", "fps", "lr", "ep_return", "ep_length", "ep_max_x", "flag_rate",
    "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "explained_var",
]


def _explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    var = np.var(target)
    return float("nan") if var == 0 else float(1 - np.var(target - pred) / var)


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                next_value: torch.Tensor, next_done: torch.Tensor,
                gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (Schulman et al., 2016).

    rewards, values, dones có shape (T, N). Quy ước: dones[t] = 1 nghĩa là quan sát tại bước t
    là khung ĐẦU của một ván mới (ván trước đã kết thúc ngay trước đó). `next_value`/`next_done`
    là V và cờ done của quan sát ngay sau bước T-1 (dùng để bootstrap).

        δ_t = r_t + γ (1 - done_{t+1}) V(s_{t+1}) - V(s_t)
        Â_t = δ_t + γ λ (1 - done_{t+1}) Â_{t+1}
        return_t = Â_t + V(s_t)        (mục tiêu của critic)
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        if t == T - 1:
            non_terminal, next_val = 1.0 - next_done, next_value
        else:
            non_terminal, next_val = 1.0 - dones[t + 1], values[t + 1]
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        adv[t] = last_gae
    return adv, adv + values


def clipped_policy_loss(ratio: torch.Tensor, adv: torch.Tensor, clip: float) -> torch.Tensor:
    """-L^CLIP của PPO: trung bình của max(-r·Â, -clip(r, 1-ε, 1+ε)·Â) (ta tối thiểu hoá nên đổi dấu)."""
    return torch.max(-adv * ratio, -adv * ratio.clamp(1 - clip, 1 + clip)).mean()


def validate_config(cfg: PPOConfig) -> None:
    parse_stages(cfg.stages)
    ScoreReward(cfg.score_weight, cfg.coin_bonus)
    if cfg.noop_max < 0 or cfg.backup_keep < 0:
        raise ValueError('noop_max and backup_keep must be nonnegative')
    if cfg.resume and cfg.init_from:
        raise ValueError('Choose resume OR init_from, not both')
    if cfg.device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    for name in ('num_envs', 'rollout_len', 'epochs', 'minibatches', 'total_steps',
                 'frame_skip', 'snapshot_every', 'backup_interval'):
        if getattr(cfg, name) <= 0:
            raise ValueError(f'{name} must be positive')
    batch_size = cfg.num_envs * cfg.rollout_len
    if batch_size % cfg.minibatches or batch_size // cfg.minibatches < 2:
        raise ValueError('Batch must divide evenly into minibatches of at least 2 samples')
    if cfg.total_steps < batch_size:
        raise ValueError('total_steps must cover at least one rollout')
    if cfg.actions not in ACTION_SETS:
        raise ValueError(f'Unknown action set: {cfg.actions}')
    if not 0.0 <= cfg.stop_flag_rate <= 1.0:
        raise ValueError('stop_flag_rate must be in [0, 1]')
    if not 1 <= cfg.stop_min_episodes <= 100:
        raise ValueError('stop_min_episodes must be in 1..100')


def stage_solved(stage_recent: dict, rate: float, min_episodes: int) -> bool:
    """True khi mọi màn đã đủ số ván gần nhất và tỉ lệ về đích đạt ngưỡng (rate > 0)."""
    if rate <= 0:
        return False
    for episodes in stage_recent.values():
        recent = list(episodes)[-min_episodes:]
        if len(recent) < min_episodes or np.mean([bool(e['flag']) for e in recent]) < rate:
            return False
    return True


def train(cfg: PPOConfig) -> Path:
    validate_config(cfg)
    run_dir = Path(cfg.out_dir) / cfg.run_name
    with run_lock(run_dir), ExitStack() as resources:
        return _train(cfg, run_dir, resources)


def _train(cfg: PPOConfig, run_dir: Path, resources: ExitStack) -> Path:
    ckpt_path = run_dir / 'ckpt_latest.pt'
    ck = None
    best_return = float('-inf')
    if cfg.resume:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f'Resume requires {ckpt_path}; existing files were preserved')
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        allowed_changes = {'resume', 'total_steps', 'out_dir', 'run_name', 'snapshot_every',
                           'drive_backup_dir', 'backup_interval', 'backup_keep', 'init_from', 'device',
                           'stop_flag_rate', 'stop_min_episodes'}
        defaults = asdict(PPOConfig())
        for name, value in asdict(cfg).items():
            if name not in allowed_changes and ck['config'].get(name, defaults[name]) != value:
                raise ValueError(f'Resume config mismatch: {name}')
        if ck['global_step'] != ck['update'] * cfg.num_envs * cfg.rollout_len:
            raise ValueError('Checkpoint step/update mismatch')
        if cfg.total_steps // (cfg.num_envs * cfg.rollout_len) <= ck['update']:
            raise ValueError('Resume requires total_steps beyond the saved update')
        best_return = restore_best_score(run_dir, ck)
    elif any(p.name != '.train.lock' for p in run_dir.iterdir()):
        raise FileExistsError(f'Run directory is not empty: {run_dir}; use resume or a new run_name')

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == 'auto' else cfg.device)

    env_kwargs = [
        dict(stages=cfg.stages, actions=cfg.actions, frame_skip=cfg.frame_skip,
             reward_scale=cfg.reward_scale, flag_bonus=cfg.flag_bonus,
             death_penalty=cfg.death_penalty, score_weight=cfg.score_weight,
             coin_bonus=cfg.coin_bonus, noop_max=cfg.noop_max, seed=cfg.seed * 1000 + i)
        for i in range(cfg.num_envs)
    ]
    n_actions = len(ACTION_SETS[cfg.actions])

    model = ActorCritic(n_actions).to(device)
    if cfg.init_from:
        source = torch.load(cfg.init_from, map_location='cpu', weights_only=True)
        transfer = initialize_policy(model, source, cfg.actions, cfg.frame_skip)
        (run_dir / 'transfer.json').write_text(json.dumps(transfer, indent=2), encoding='utf-8')
        print(f'[warm start] {transfer}', flush=True)
        del source
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    T, N = cfg.rollout_len, cfg.num_envs
    batch_size = T * N
    minibatch_size = batch_size // cfg.minibatches
    num_updates = cfg.total_steps // batch_size

    # --- bộ đệm rollout nằm trên GPU
    obs_buf = torch.zeros((T, N, 4, 84, 84), dtype=torch.uint8, device=device)
    act_buf = torch.zeros((T, N), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, N), device=device)
    rew_buf = torch.zeros((T, N), device=device)
    done_buf = torch.zeros((T, N), device=device)
    val_buf = torch.zeros((T, N), device=device)

    global_step, start_update = 0, 1
    if ck is not None:
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        global_step, start_update = ck["global_step"], ck["update"] + 1
        print(f"[resume] tiếp tục từ update {start_update - 1}, step {global_step:,}", flush=True)
        reconcile_logs(run_dir, global_step)
    (run_dir / 'config.json').write_text(json.dumps(asdict(cfg), indent=2), encoding='utf-8')

    metrics_path, episodes_path = run_dir / "metrics.csv", run_dir / "episodes.csv"
    metrics_header = not metrics_path.exists() or metrics_path.stat().st_size == 0
    episodes_header = not episodes_path.exists() or episodes_path.stat().st_size == 0
    episode_fields = ['global_step', 'stage', 'return', 'length', 'max_x', 'flag', 'time_left',
                      'score', 'score_gain', 'coins_collected', 'base_return', 'score_reward_raw']
    if not episodes_header:
        with episodes_path.open(newline='', encoding='utf-8') as existing:
            episode_fields = next(csv.reader(existing))  # preserve legacy resume CSV schema
    mf = resources.enter_context(open(metrics_path, 'a', newline='', encoding='utf-8'))
    ef = resources.enter_context(open(episodes_path, 'a', newline='', encoding='utf-8'))
    mw, ew = csv.DictWriter(mf, METRIC_FIELDS), csv.DictWriter(ef, episode_fields, extrasaction='ignore')
    if metrics_header:
        mw.writeheader()
    if episodes_header:
        ew.writeheader()

    recent = deque(ck.get('recent', []) if ck else [], maxlen=100)
    stage_recent = {f'{w}-{s}': deque(maxlen=100) for w, s in parse_stages(cfg.stages)}
    for stage, episodes in (ck.get('stage_recent', {}) if ck else {}).items():
        if stage in stage_recent:
            stage_recent[stage].extend(episodes)
    next_snapshot = (global_step // cfg.snapshot_every + 1) * cfg.snapshot_every

    envs = SubprocVecEnv(env_kwargs)
    resources.callback(envs.close)
    next_obs = torch.as_tensor(envs.reset(), device=device)
    next_done = torch.zeros(N, device=device)
    t_start, step_at_start = time.time(), global_step
    last_backup = float('-inf')

    try:
        for update in range(start_update, num_updates + 1):
            lr = cfg.lr * (1.0 - (update - 1) / num_updates) if cfg.anneal_lr else cfg.lr
            for g in optimizer.param_groups:
                g["lr"] = lr

            # ---------------------------------------------------- 1) thu thập rollout
            for t in range(T):
                global_step += N
                obs_buf[t], done_buf[t] = next_obs, next_done
                with torch.no_grad():
                    action, logp, _, value = model.get_action_and_value(next_obs)
                act_buf[t], logp_buf[t], val_buf[t] = action, logp, value

                obs_np, reward, done, episodes = envs.step(action.cpu().numpy())
                rew_buf[t] = torch.as_tensor(reward, device=device)
                next_obs = torch.as_tensor(obs_np, device=device)
                next_done = torch.as_tensor(done, dtype=torch.float32, device=device)
                for ep in episodes:
                    if ep is None:
                        continue
                    recent.append(ep)
                    stage_recent[ep['stage']].append(ep)
                    ew.writerow({**ep, 'global_step': global_step, 'return': f"{ep['return']:.3f}",
                                 'flag': int(ep['flag'])})

            # ---------------------------------------------------- 2) GAE
            with torch.no_grad():
                next_value = model.get_value(next_obs)
                adv, returns = compute_gae(rew_buf, val_buf, done_buf, next_value, next_done,
                                           cfg.gamma, cfg.gae_lambda)

            b_obs = obs_buf.reshape(-1, 4, 84, 84)
            b_act, b_logp = act_buf.reshape(-1), logp_buf.reshape(-1)
            b_adv, b_ret, b_val = adv.reshape(-1), returns.reshape(-1), val_buf.reshape(-1)

            # ---------------------------------------------------- 3) tối ưu hoá
            inds = np.arange(batch_size)
            clip_fracs, pg_losses, v_losses, entropies, kls = [], [], [], [], []
            stop = False
            for _ in range(cfg.epochs):
                np.random.shuffle(inds)
                for s in range(0, batch_size, minibatch_size):
                    mb = torch.as_tensor(inds[s:s + minibatch_size], device=device)
                    _, new_logp, entropy, new_val = model.get_action_and_value(b_obs[mb], b_act[mb])
                    log_ratio = new_logp - b_logp[mb]
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clip_fracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                    mb_adv = b_adv[mb]
                    if cfg.norm_adv:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    pg_loss = clipped_policy_loss(ratio, mb_adv, cfg.clip_coef)

                    if cfg.clip_vloss:
                        v_unclipped = (new_val - b_ret[mb]) ** 2
                        v_clipped = b_val[mb] + (new_val - b_val[mb]).clamp(-cfg.clip_coef, cfg.clip_coef)
                        v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_ret[mb]) ** 2).mean()
                    else:
                        v_loss = 0.5 * ((new_val - b_ret[mb]) ** 2).mean()

                    ent = entropy.mean()
                    loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                    pg_losses.append(pg_loss.item()); v_losses.append(v_loss.item())
                    entropies.append(ent.item()); kls.append(approx_kl.item())
                if cfg.target_kl is not None and np.mean(kls[-cfg.minibatches:]) > cfg.target_kl:
                    stop = True
                if stop:
                    break

            # ---------------------------------------------------- 4) ghi log + checkpoint
            fps = (global_step - step_at_start) / (time.time() - t_start)
            row = dict(
                update=update, global_step=global_step, fps=round(fps), lr=lr,
                ep_return=np.mean([e["return"] for e in recent]) if recent else float("nan"),
                ep_length=np.mean([e["length"] for e in recent]) if recent else float("nan"),
                ep_max_x=np.mean([e["max_x"] for e in recent]) if recent else float("nan"),
                flag_rate=np.mean([e["flag"] for e in recent]) if recent else float("nan"),
                policy_loss=np.mean(pg_losses), value_loss=np.mean(v_losses),
                entropy=np.mean(entropies), approx_kl=np.mean(kls), clip_frac=np.mean(clip_fracs),
                explained_var=_explained_variance(b_val.cpu().numpy(), b_ret.cpu().numpy()),
            )
            mw.writerow(row); mf.flush(); ef.flush()
            if update % 10 == 0 or update == num_updates:
                by_stage = {}
                for stage, episodes in stage_recent.items():
                    by_stage[stage] = {'episodes': len(episodes)}
                    for key in ('flag', 'max_x', 'score_gain', 'coins_collected'):
                        values = [e[key] for e in episodes if key in e]
                        by_stage[stage][key + '_mean'] = float(np.mean(values)) if values else None
                (run_dir / 'stage_metrics.json').write_text(
                    json.dumps({'global_step': global_step, 'stages': by_stage}, indent=2), encoding='utf-8')
            if update % 5 == 0 or update == 1:
                print(f"upd {update:5d}/{num_updates} | step {global_step:>10,} | {fps:5.0f} sps | "
                      f"ret {row['ep_return']:7.2f} | x {row['ep_max_x']:6.0f} | flag {row['flag_rate']:.2f} | "
                      f"ent {row['entropy']:.3f} | kl {row['approx_kl']:.4f} | ev {row['explained_var']:.2f}",
                      flush=True)

            improved = len(recent) >= 50 and row['ep_return'] > best_return
            if improved:
                best_return = float(row['ep_return'])
            solved = stage_solved(stage_recent, cfg.stop_flag_rate, cfg.stop_min_episodes)
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         global_step=global_step, update=update, config=asdict(cfg),
                         best_return=best_return, recent=list(recent),
                         stage_recent={s: list(es) for s, es in stage_recent.items()},
                         stopped_early=solved)
            if update % 10 == 0 or update == num_updates or solved:
                atomic_save(state, ckpt_path)
            if improved:
                atomic_save(state, run_dir / "ckpt_best.pt")
            if global_step >= next_snapshot:
                atomic_save(state, run_dir / f"ckpt_{global_step // 1000}k.pt")
                next_snapshot += cfg.snapshot_every
            if cfg.drive_backup_dir and time.monotonic() - last_backup >= cfg.backup_interval:
                atomic_save(state, ckpt_path)
                saved = backup_run(run_dir, cfg.drive_backup_dir, global_step, cfg.backup_keep)
                last_backup = time.monotonic()
                print(f'[Drive backup] {saved}', flush=True)
            if solved:
                print(f'[dừng sớm] {cfg.stop_min_episodes} ván gần nhất của mọi màn về đích '
                      f'>= {cfg.stop_flag_rate:.0%} tại step {global_step:,}', flush=True)
                break
    finally:
        mf.flush(); ef.flush()

    atomic_save(state, run_dir / "ckpt_final.pt")
    if cfg.drive_backup_dir:
        saved = backup_run(run_dir, cfg.drive_backup_dir, global_step, cfg.backup_keep)
        print(f'[Drive backup final] {saved}', flush=True)
    print(f"Xong. Đã lưu vào {run_dir}", flush=True)
    return run_dir
