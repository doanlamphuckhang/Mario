"""Huấn luyện agent PPO chơi Mario.

Ví dụ:
    python train.py --run_name ppo_1-1 --stages 1-1 --total_steps 5000000
    python train.py --run_name smoke --total_steps 20000 --num_envs 4     # chạy thử nhanh
"""
import argparse
import dataclasses
import json
from pathlib import Path
import sys
import typing

from mario_rl.ppo import PPOConfig, train


def bool_arg(value):
    if isinstance(value, bool):
        return value
    if value.lower() not in ('1', '0', 'true', 'false', 'yes', 'no'):
        raise argparse.ArgumentTypeError('Expected true/false')
    return value.lower() in ('1', 'true', 'yes')


def build_parser(defaults=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config', help='JSON experiment profile; explicit flags override it')
    hints = typing.get_type_hints(PPOConfig)
    for f in dataclasses.fields(PPOConfig):
        t = hints[f.name]
        if t is bool:
            parser.add_argument(f"--{f.name}", type=bool_arg,
                                default=f.default, metavar="BOOL")
        elif t == (float | None):
            parser.add_argument(f"--{f.name}", type=float, default=f.default)
        else:
            parser.add_argument(f"--{f.name}", type=t, default=f.default)
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def parse_config(argv=None):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config')
    known, _ = probe.parse_known_args(argv)
    defaults = {}
    if known.config:
        defaults = json.loads(Path(known.config).read_text(encoding='utf-8-sig'))
        if not isinstance(defaults, dict):
            raise ValueError('Config must be a JSON object')
        hints = typing.get_type_hints(PPOConfig)
        unknown = set(defaults) - set(hints)
        if unknown:
            raise ValueError(f'Unknown config fields: {sorted(unknown)}')
        for name, value in defaults.items():
            t = hints[name]
            valid = (value is None or type(value) in (int, float)) if t == (float | None) else (
                type(value) in (int, float) if t is float else type(value) is t)
            if not valid:
                raise ValueError(f'Invalid type for config field {name}')
    values = vars(build_parser(defaults).parse_args(argv))
    values.pop('config')
    return PPOConfig(**values)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = parse_config()
    train(cfg)


if __name__ == '__main__':
    main()
