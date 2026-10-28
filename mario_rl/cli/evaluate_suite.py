"""Evaluate checkpoints on the SAME stage/seed/mode matrix and save raw episodes.

No training, downloads or automatic GPU use. Run separately after training ends.
Score gain is pre-flag gameplay score, not an enemy kill counter. Return depends
on the checkpoint's reward configuration and must not rank different objectives.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from mario_rl.envs import parse_stages
from mario_rl.evaluation import load_model, run_eval_vec, summarize


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpts', nargs='+', required=True)
    p.add_argument('--stages', default='1-1,1-2,1-3,1-4,2-1,2-2,2-3,2-4')
    p.add_argument('--seeds', default='12345,23456,34567')
    p.add_argument('--episodes', type=int, default=20, help='per stage, seed and action mode')
    p.add_argument('--modes', choices=['both', 'argmax', 'sample'], default='both')
    p.add_argument('--noop_max', type=int, default=30)
    p.add_argument('--num_envs', type=int, default=4)
    p.add_argument('--max_steps', type=int, default=6000)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--out', required=True, help='new directory; existing results are never overwritten')
    return p


def run_suite(args):
    stages = [f'{w}-{s}' for w, s in parse_stages(args.stages)]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError('Use distinct nonnegative evaluation seeds')
    if min(args.episodes, args.num_envs, args.max_steps) <= 0 or args.noop_max < 0:
        raise ValueError('Invalid evaluation limits')
    checkpoints = [Path(p).resolve(strict=True) for p in args.ckpts]
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError('Duplicate checkpoint')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'protocol.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    modes = ['argmax', 'sample'] if args.modes == 'both' else [args.modes]
    device = torch.device(args.device)
    torch.set_num_threads(4)
    with (out / 'episodes.jsonl').open('x', encoding='utf-8') as details, \
            (out / 'summary.csv').open('x', newline='', encoding='utf-8') as table:
        writer = None
        for checkpoint in checkpoints:
            with checkpoint.open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            model, cfg = load_model(checkpoint, device)
            trained = {f'{w}-{s}' for w, s in parse_stages(cfg['stages'])}
            for stage in stages:
                for seed in seeds:
                    for mode in modes:
                        identity = dict(checkpoint=str(checkpoint), sha256=digest, stage=stage,
                                        trained_stage=stage in trained, seed=seed, mode=mode,
                                        noop_max=args.noop_max)
                        print(json.dumps(identity), flush=True)
                        try:
                            episodes = run_eval_vec(model, cfg, stage, args.episodes,
                                                    mode == 'argmax', args.noop_max, seed, device,
                                                    args.num_envs, args.max_steps)
                        except Exception as exc:
                            details.write(json.dumps({**identity, 'error': str(exc)}) + '\n')
                            details.flush()
                            raise  # no fabricated zero score or automatic retry
                        details.write(json.dumps({**identity, 'reward_config': cfg,
                                                  'episodes': episodes}) + '\n')
                        details.flush()
                        row = {**identity, **summarize(episodes)}
                        if writer is None:
                            writer = csv.DictWriter(table, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        table.flush()
                        print(f"  wins={row['flag_rate']:.1%}, score_gain={row['score_gain_mean']}, "
                              f"coins={row['coins_collected_mean']}", flush=True)
            del model
    return out


def main(argv=None):
    return run_suite(build_parser().parse_args(argv))


if __name__ == '__main__':
    main()
