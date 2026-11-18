"""声源测试：周期性、相位连续、扫描支持。"""

import numpy as np

from vowel_synth.source import GlottalSource


def test_rosenberg_period_matches_f0():
    fs = 22050
    src = GlottalSource(fs, "rosenberg")
    f0 = 120.0
    x = src.generate_block(fs, f0, open_quotient=0.6, noise_gain=0.0)
    # 自相关主峰应出现在一个基频周期处
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[x.size - 1:]
    lag_min, lag_max = int(fs / (f0 * 1.2)), int(fs / (f0 * 0.8))
    period = lag_min + np.argmax(ac[lag_min:lag_max])
    assert abs(fs / period - f0) / f0 < 0.02


def test_phase_continuity_across_blocks():
    fs = 22050
    src = GlottalSource(fs)
    a = src.generate_block(1000, 150.0)
    b = src.generate_block(1000, 150.0)
    # 一次性生成同样长度，拼接处不应出现额外跳变
    src2 = GlottalSource(fs)
    whole = src2.generate_block(2000, 150.0)
    # 分块与整块生成仅差浮点累加顺序，容差放到 1e-10
    np.testing.assert_allclose(np.concatenate([a, b]), whole, atol=1e-10)


def test_sweep_phase_never_jumps():
    fs = 22050
    src = GlottalSource(fs)
    f0 = np.linspace(80, 400, 2048)
    x = src.generate_block(2048, f0)
    assert np.all(np.isfinite(x))
    # 相邻样本差分有界（无相位回绕造成的大幅尖峰）
    assert np.max(np.abs(np.diff(x))) < 20.0


def test_noise_component_independent_and_mixes():
    fs = 22050
    src = GlottalSource(fs)
    voiced = src.generate_block(4096, 120.0, noise_gain=0.0)
    src.reset()
    mixed = src.generate_block(4096, 120.0, noise_gain=0.5)
    assert not np.allclose(voiced, mixed)
    # 混合信号能量更高
    assert np.mean(mixed ** 2) > np.mean(voiced ** 2)


def test_saw_shape_runs():
    src = GlottalSource(22050, "saw")
    x = src.generate_block(2048, 100.0)
    assert x.min() < -0.9 and x.max() > 0.9
