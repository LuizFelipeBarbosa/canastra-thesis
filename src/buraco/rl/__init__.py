"""PPO self-play training on top of BuracoEnv (torch lives only in this package).

`buraco.rl.obs` and `buraco.rl.buffer` are numpy-only; everything importing
torch does so lazily enough that the engine/env never depend on it.
"""
