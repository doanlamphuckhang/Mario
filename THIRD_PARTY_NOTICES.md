# Third-party attribution

`mario_rl/score_reward.py` adapts score-delta reward shaping from
`CustomReward.step` in Viet Nguyen's Super-mario-bros-PPO-pytorch:
https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch/blob/ab4248d715346c6adc33c2157455e2b98c130bcc/src/env.py

Changes: configurable coefficient (upstream uses 1/40), optional coin bonus,
coin counter rollover, reset baseline, pre-flag score accounting, episode totals.
The upstream trainer, stage-coordinate rules and weights are not copied.

MIT License

Copyright (c) 2020 Viet Nguyen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
