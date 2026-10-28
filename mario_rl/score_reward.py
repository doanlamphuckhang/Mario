"""Score-delta shaping adapted from Viet Nguyen's MIT-licensed CustomReward.

Source and license: THIRD_PARTY_NOTICES.md. Score includes enemies, coins,
blocks and items; it is deliberately NOT reported as an enemy kill count.
All weights below are in raw reward units, before reward_scale.
"""
from math import isfinite


class ScoreReward:
    def __init__(self, score_weight=0.0, coin_bonus=0.0):
        if not all(isfinite(x) and x >= 0 for x in (score_weight, coin_bonus)):
            raise ValueError('Score/coin reward weights must be finite and nonnegative')
        self.score_weight = score_weight
        self.coin_bonus = coin_bonus
        self.reset()

    def reset(self, score=0, coins=0):
        self.previous_score = int(score)
        self.previous_coins = int(coins)
        self.score_gain = 0
        self.coins_collected = 0
        self.reward_total = 0.0
        self.finished = False

    def step(self, info):
        score = int(info.get('score', self.previous_score))
        coins = int(info.get('coins', self.previous_coins))
        # HUD coins wrap 99 -> 0 after granting a life. Called per emulator step.
        coin_delta = (coins - self.previous_coins) % 100
        score_delta = max(0, score - self.previous_score)
        self.previous_score, self.previous_coins = score, coins
        # Exclude the flag frame and subsequent end-of-level score/timer bonuses.
        self.finished = self.finished or bool(info.get('flag_get', False))
        if self.finished:
            return 0.0
        self.score_gain += score_delta
        self.coins_collected += coin_delta
        reward = self.score_weight * score_delta + self.coin_bonus * coin_delta
        self.reward_total += reward
        return reward
