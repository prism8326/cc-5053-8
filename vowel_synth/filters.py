"""共振峰滤波器：二阶谐振器级联。

每个共振峰是一个带单位共振增益的双极点谐振器（Klatt 1980 形式）：
    y[n] = C*x[n] + a1*y[n-1] + a2*y[n-2]
    theta = 2*pi*f/fs,  R = exp(-pi*bw/fs)
    a1 = 2 R cos(theta),  a2 = -R^2
    C  = (1-R) * sqrt(1 - 2 R cos(2 theta) + R^2)   # 共振处增益归一化为 1

设计要点：
- 中心频率 f 与基频（音高）完全无关，f 只由声道配置决定。
- 所有系数在“支持范围”内解析地保证稳定（极点半径 R < 1）。
- 越界频率与过窄带宽不抛异常、不静默，而是钳制到边界并记录告警，
  界面与调用方可以展示“请求值/实际生效值”。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

# 极点半径上限：留数值裕量，避免 R→1 时自激和长时间振铃。
R_MAX = 0.999
# 频率离 Nyquist 的最小相对裕量（theta 不得逼近 pi）。
NYQUIST_MARGIN = 0.02
# 可表示的最低共振频率（Hz），避免 theta≈0 的数值退化。
F_MIN = 10.0


@dataclass
class ResonatorSpec:
    """单个共振峰的系数与钳制记录。"""
    f_requested: float
    bw_requested: float
    f_effective: float
    bw_effective: float
    fs: float
    C: float
    a1: float
    a2: float
    warnings: list[str] = field(default_factory=list)

    @property
    def stable(self) -> bool:
        # 对该二阶节，|a2| = R^2 < 1 即稳定。
        return abs(self.a2) < 1.0

    def ba(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([self.C, 0.0, 0.0]), np.array([1.0, -self.a1, -self.a2])


def design_resonator(f: float, bw: float, fs: float) -> ResonatorSpec:
    """带边界处理的谐振器设计。

    - f 钳制到 [F_MIN, fs/2*(1-NYQUIST_MARGIN)]
    - bw 钳制到使 R <= R_MAX 的最小值（过窄带宽被展宽）
    - bw 同时钳制非负；任何钳制都写入 warnings
    """
    warns: list[str] = []
    f_req, bw_req = float(f), float(bw)

    f_hi = 0.5 * fs * (1.0 - NYQUIST_MARGIN)
    f_eff = min(max(f_req, F_MIN), f_hi)
    if f_req < F_MIN:
        warns.append(f"f={f_req:.1f}Hz below floor, clamped to {F_MIN:.1f}Hz")
    elif f_req > f_hi:
        warns.append(f"f={f_req:.1f}Hz above Nyquist-safe limit, "
                     f"clamped to {f_hi:.1f}Hz")

    # R = exp(-pi*bw/fs)；R<=R_MAX 等价于 bw >= -fs/pi*ln(R_MAX)
    bw_min = -fs / np.pi * np.log(R_MAX)
    bw_eff = max(bw_req, bw_min)
    if bw_req < bw_min:
        warns.append(
            f"bw={bw_req:.1f}Hz too narrow (pole radius > {R_MAX}), "
            f"widened to {bw_min:.1f}Hz")
    if not np.isfinite(bw_req) or bw_req <= 0:
        warns.append(f"bw={bw_req!r} non-positive, using minimum "
                     f"{bw_min:.1f}Hz")

    theta = 2.0 * np.pi * f_eff / fs
    R = np.exp(-np.pi * bw_eff / fs)
    R = min(R, R_MAX)
    a1 = 2.0 * R * np.cos(theta)
    a2 = -(R ** 2)
    C = (1.0 - R) * np.sqrt(max(0.0, 1.0 - 2.0 * R * np.cos(2.0 * theta) + R ** 2))

    return ResonatorSpec(f_req, bw_req, f_eff, bw_eff, fs,
                         float(C), float(a1), float(a2), warns)


class FormantCascade:
    """若干谐振器的级联，块间保持滤波器状态，支持逐块更新系数。

    系数在块边界切换（零阶保持）。平滑参数变化由调用方在块间做插值，
    本类只负责保证任意时刻的系数都稳定。
    """

    def __init__(self, n_formants: int, fs: float):
        self.n_formants = n_formants
        self.fs = fs
        self._zi = [np.zeros(2) for _ in range(n_formants)]
        self.specs: list[ResonatorSpec] = []

    def reset(self) -> None:
        self._zi = [np.zeros(2) for _ in range(self.n_formants)]

    def process_block(self, x: np.ndarray,
                      freqs: np.ndarray, bws: np.ndarray
                      ) -> tuple[np.ndarray, list[str]]:
        """对一个块依次通过全部谐振器，返回 (输出, 本块告警列表)。"""
        freqs = np.asarray(freqs, dtype=float).reshape(-1)
        bws = np.asarray(bws, dtype=float).reshape(-1)
        if freqs.size != self.n_formants or bws.size != self.n_formants:
            raise ValueError(
                f"需要 {self.n_formants} 个共振峰参数，收到 f:{freqs.size} bw:{bws.size}")

        y = np.asarray(x, dtype=float).copy()
        all_warns: list[str] = []
        self.specs = []
        for k in range(self.n_formants):
            spec = design_resonator(freqs[k], bws[k], self.fs)
            self.specs.append(spec)
            if spec.warnings:
                all_warns.extend([f"F{k + 1}: {w}" for w in spec.warnings])
            b = np.array([spec.C, 0.0, 0.0])
            a = np.array([1.0, -spec.a1, -spec.a2])
            y, self._zi[k] = lfilter(b, a, y, zi=self._zi[k])
        return y, all_warns

    def frequency_response(self, freqs: np.ndarray, bws: np.ndarray,
                           n_fft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """当前参数下整层级联的频率响应，返回 (频率轴Hz, 幅度dB)。"""
        w = np.linspace(0.0, np.pi, n_fft)
        H = np.ones(n_fft, dtype=complex)
        for f, bw in zip(np.asarray(freqs, float), np.asarray(bws, float)):
            spec = design_resonator(f, bw, self.fs)
            z = np.exp(1j * w)
            H *= spec.C / (1.0 - spec.a1 * z ** -1 - spec.a2 * z ** -2)
        return w * self.fs / (2.0 * np.pi), 20.0 * np.log10(np.abs(H) + 1e-12)
