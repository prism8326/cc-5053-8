"""周期声源 + 噪声分量。

声门源与声道在参数上完全解耦：本模块只负责产生激励，
共振峰由 filters.FormantCascade 单独塑造。

两种周期源：
- rosenberg：分段平滑的 Rosenberg 型声门流**导数**，
  开相 0→1 平滑上升、闭相 1→0 平滑回落，谱随 OQ 可控倾斜；
- saw：锯齿波（脉冲串的积分），谐波近似等幅，便于频谱教学。

基频 f0 由相位累加器积分，支持逐样本变化（音高扫描），
块与块之间相位连续，扫描不会产生跳变咔哒声。
"""

from __future__ import annotations

import numpy as np


class GlottalSource:
    def __init__(self, fs: float, shape: str = "rosenberg"):
        self.fs = fs
        self.shape = shape
        self._phase = 0.0  # 归一化相位 [0, 1)
        self.rng = np.random.default_rng(0)

    def reset(self) -> None:
        self._phase = 0.0

    def _advance_phase(self, n: int, f0: np.ndarray) -> np.ndarray:
        """返回 n 个样本的周期内相位，并推进内部相位。"""
        f0 = np.broadcast_to(np.asarray(f0, dtype=float), (n,))
        inc = f0 / self.fs
        phases = (self._phase + np.cumsum(inc) - inc[0]) % 1.0
        self._phase = (self._phase + float(np.sum(inc))) % 1.0
        return phases

    def _periodic(self, phases: np.ndarray, open_quotient: float,
                  f0: np.ndarray) -> np.ndarray:
        oq = float(np.clip(open_quotient, 0.1, 0.95))
        if self.shape == "saw":
            # 锯齿：每周期 0→1 线性上升后复位（相位导数恒定）
            return 2.0 * phases - 1.0

        # Rosenberg 型声门流，对相位的解析导数（即声门激励）
        d = np.empty_like(phases)
        open_mask = phases < oq
        po = phases[open_mask]
        pc = phases[~open_mask]
        # 开相 g=0.5*(1-cos(pi p/oq))  → g'=0.5*sin(pi p/oq)*pi/oq
        d[open_mask] = 0.5 * np.sin(np.pi * po / oq) * (np.pi / oq)
        # 闭相 g=0.5*(1+cos(pi (p-oq)/(1-oq))) → 负斜率回落
        d[~open_mask] = (-0.5 * np.sin(np.pi * (pc - oq) / (1.0 - oq))
                         * (np.pi / (1.0 - oq)))
        # 链法则：dg/dt = dg/dp * f0；扫描时按当前样本 f0 缩放
        f0_scale = np.broadcast_to(np.asarray(f0, float) / 120.0, phases.shape)
        return d * f0_scale

    def generate_block(self, n: int, f0: float | np.ndarray,
                       open_quotient: float = 0.6,
                       noise_gain: float = 0.0,
                       voiced_gain: float = 1.0) -> np.ndarray:
        """生成 n 个激励样本。f0 可为标量或长度 n 的数组（扫描）。"""
        f0_arr = np.full(n, float(f0)) if np.isscalar(f0) else np.asarray(f0, float)
        phases = self._advance_phase(n, f0_arr)
        voiced = self._periodic(phases, open_quotient, f0_arr)
        # 独立噪声分量（送气声），在进入共振峰级联之前混合
        noise = self.rng.standard_normal(n)
        return voiced_gain * voiced + noise_gain * noise
