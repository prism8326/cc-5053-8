"""合成引擎：声源 → 共振峰级联 → 输出。

关键约定：f0（音高）只驱动声源的相位累加，绝不修改共振峰频率。
改变音高时谐波间距变化、共振包络不动。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .filters import FormantCascade
from .source import GlottalSource


@dataclass
class SynthParams:
    f0: float = 120.0                 # 基频 Hz（音高）
    freqs: np.ndarray = field(        # 各共振峰中心频率 Hz
        default_factory=lambda: np.array([730., 1090., 2440., 3500.]))
    bws: np.ndarray = field(          # 各共振峰带宽 Hz
        default_factory=lambda: np.array([80., 90., 120., 180.]))
    open_quotient: float = 0.6        # 声门开相占比（声源形状）
    shape: str = "rosenberg"          # rosenberg | saw
    noise_gain: float = 0.02          # 送气噪声分量
    voiced_gain: float = 1.0
    tilt_db: float = 0.0              # 源谱倾斜：高频相对低频增益(dB)

    def __post_init__(self) -> None:
        self.freqs = np.asarray(self.freqs, dtype=float)
        self.bws = np.asarray(self.bws, dtype=float)
        if self.freqs.shape != self.bws.shape:
            raise ValueError("freqs 与 bws 长度必须一致")

    @property
    def n_formants(self) -> int:
        return self.freqs.size


def _as_envelope(value, n_blocks: int) -> np.ndarray:
    """标量→常量包络；数组→必须长度 n_blocks 的逐块包络。"""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_blocks, float(arr))
    if arr.size != n_blocks:
        raise ValueError(f"包络长度 {arr.size} != 块数 {n_blocks}")
    return arr.astype(float)


def render(duration: float, fs: float, params: SynthParams,
           block_size: int = 256, f0_envelope: np.ndarray | None = None,
           freq_envelope: np.ndarray | None = None,
           bw_envelope: np.ndarray | None = None,
           tilt_envelope: np.ndarray | None = None,
           noise_envelope: np.ndarray | None = None,
           voiced_envelope: np.ndarray | None = None,
           normalize: bool = True, seed: int = 0) -> dict:
    """离线渲染。包络可选，按块给出；未给则使用 params 中的常量。

    返回 dict：y(输出), source(源信号), f0_track, freqs, bws,
    tilt, noise, voiced（逐块包络），effective(每块实际系数), warnings。
    """
    n = int(round(duration * fs))
    n_blocks = int(np.ceil(n / block_size))
    nf = params.n_formants

    f0_env = _as_envelope(f0_envelope if f0_envelope is not None else params.f0,
                          n_blocks)
    tilt_env = _as_envelope(
        tilt_envelope if tilt_envelope is not None else params.tilt_db, n_blocks)
    noise_env = _as_envelope(
        noise_envelope if noise_envelope is not None else params.noise_gain,
        n_blocks)
    voiced_env = _as_envelope(
        voiced_envelope if voiced_envelope is not None else params.voiced_gain,
        n_blocks)
    if freq_envelope is None:
        freq_env = np.tile(params.freqs, (n_blocks, 1))
    else:
        freq_env = np.asarray(freq_envelope, float)
        if freq_env.shape != (n_blocks, nf):
            raise ValueError(f"freq_envelope 形状须为 ({n_blocks},{nf})")
    if bw_envelope is None:
        bw_env = np.tile(params.bws, (n_blocks, 1))
    else:
        bw_env = np.asarray(bw_envelope, float)
        if bw_env.shape != (n_blocks, nf):
            raise ValueError(f"bw_envelope 形状须为 ({n_blocks},{nf})")

    source = GlottalSource(fs, params.shape)
    source.rng = np.random.default_rng(seed)
    cascade = FormantCascade(nf, fs)

    y_out = np.zeros(n)
    src_out = np.zeros(n)
    effective, warnings = [], []
    for b in range(n_blocks):
        i0, i1 = b * block_size, min(n, (b + 1) * block_size)
        m = i1 - i0
        # 块内 f0 线性插值，扫描平滑
        f0_prev = f0_env[b - 1] if b else f0_env[0]
        f0_block = f0_prev + (f0_env[b] - f0_prev) * np.linspace(0, 1, m, endpoint=False)
        # 源谱倾斜同样线性过渡，避免块边界出现音色台阶
        tilt_prev = tilt_env[b - 1] if b else tilt_env[0]
        tilt_block = tilt_prev + (tilt_env[b] - tilt_prev) * np.linspace(
            0, 1, m, endpoint=False)

        src = source.generate_block(
            m, f0_block, open_quotient=params.open_quotient,
            noise_gain=float(noise_env[b]), voiced_gain=float(voiced_env[b]),
            tilt_db=tilt_block)
        y, warns = cascade.process_block(src, freq_env[b], bw_env[b])
        src_out[i0:i1], y_out[i0:i1] = src, y
        if warns:
            warnings.append({"block": b, "messages": warns})
        effective.append([(s.f_effective, s.bw_effective, s.C, s.a1, s.a2)
                          for s in cascade.specs])

    if normalize:
        peak = np.max(np.abs(y_out))
        if peak > 0:
            y_out = 0.9 * y_out / peak
    else:
        y_out = np.tanh(y_out)  # 软限幅，防止极端参数削波
    return {
        "y": y_out,
        "source": src_out,
        "fs": fs,
        "f0_track": np.repeat(f0_env, block_size)[:n],
        "freqs": freq_env,
        "bws": bw_env,
        "tilt": tilt_env,
        "noise": noise_env,
        "voiced": voiced_env,
        "block_size": block_size,
        "effective": np.array(effective),   # (n_blocks, n_formants, 5)
        "warnings": warnings,
    }


class RealtimeEngine:
    """低延迟实时块处理器，系数在块边界更新且始终稳定。"""

    def __init__(self, fs: float, params: SynthParams, block_size: int = 256):
        self.fs = fs
        self.block_size = block_size
        self.params = params
        self._source = GlottalSource(fs, params.shape)
        self._cascade = FormantCascade(params.n_formants, fs)
        self.last_warnings: list[str] = []
        self.last_specs: list = []

    def update_params(self, params: SynthParams) -> None:
        if params.n_formants != self._cascade.n_formants:
            raise ValueError("共振峰数量不能在运行中改变")
        self.params = params
        self._source.shape = params.shape

    def process_block(self, block_size: int | None = None) -> np.ndarray:
        m = block_size or self.block_size
        p = self.params
        src = self._source.generate_block(
            m, p.f0, open_quotient=p.open_quotient,
            noise_gain=p.noise_gain, voiced_gain=p.voiced_gain,
            tilt_db=p.tilt_db)
        y, warns = self._cascade.process_block(src, p.freqs, p.bws)
        self.last_warnings, self.last_specs = warns, self._cascade.specs
        return np.tanh(y)
