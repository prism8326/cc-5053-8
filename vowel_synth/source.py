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
from scipy.signal import lfilter

# 源谱倾斜滤波器的枢轴频率与增益范围。倾斜量以“高频相对低频的增益(dB)”
# 表示：0 dB 为直通，负值变暗，正值变亮。滤波器是稳定一阶低通（两级级联）
# 与直通的线性混合，任何倾斜值都不会改变极点位置。
TILT_PIVOT_HZ = 1000.0
TILT_GAIN_MIN = 0.02
TILT_GAIN_MAX = 6.0
TILT_ORDER = 2


class SpectralTilt:
    """无源共振的源谱倾斜层。

    y = (1-m)*x + m*lowpass_order2(x)，lowpass 在枢轴频率处为 -3 dB/级，
    直流增益 1。m=0 完全直通；m>0 变暗，m<0 变亮。极点由枢轴固定，
    倾斜只改变混合量，不会产生新的谱峰。
    """

    def __init__(self, fs: float, pivot_hz: float = TILT_PIVOT_HZ,
                 order: int = TILT_ORDER):
        self.fs = fs
        self.pivot_hz = pivot_hz
        self.order = max(1, int(order))
        self._p = float(np.exp(-2.0 * np.pi * pivot_hz / fs))
        self._zi = np.zeros(self.order)

    def reset(self) -> None:
        self._zi = np.zeros(self.order)

    def _lowpass_nyquist_gain(self) -> float:
        return ((1.0 - self._p) / (1.0 + self._p)) ** self.order

    def mix_for_db(self, tilt_db: float | np.ndarray) -> np.ndarray:
        """把高频相对增益(dB)映射为低通混合量 m。

        一阶节在 DC 的增益为 1、在 Nyquist 的增益为 L；级联后仍为 L^n。
        解 (1-m) + m*L^n = g 得 m = (1-g)/(1-L^n)。
        """
        gain = np.clip(10.0 ** (np.asarray(tilt_db, float) / 20.0),
                       TILT_GAIN_MIN, TILT_GAIN_MAX)
        l_nyq = self._lowpass_nyquist_gain()
        return (1.0 - gain) / (1.0 - l_nyq)

    def process(self, x: np.ndarray,
                mix: float | np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        lp = x
        for k in range(self.order):
            lp, self._zi[k:k + 1] = lfilter(
                [1.0 - self._p], [1.0, -self._p], lp,
                zi=self._zi[k:k + 1])
        m = np.broadcast_to(np.asarray(mix, float), x.shape)
        return (1.0 - m) * x + m * lp

    def response_db(self, freqs: np.ndarray,
                    tilt_db: float) -> np.ndarray:
        """解析频率响应，用于核验“倾斜只改斜率、不造峰”。"""
        w = 2.0 * np.pi * np.asarray(freqs, float) / self.fs
        m = float(self.mix_for_db(tilt_db))
        z = np.exp(-1j * w)
        L = (1.0 - self._p) / (1.0 - self._p * z)
        H = (1.0 - m) + m * L ** self.order
        return 20.0 * np.log10(np.abs(H) + 1e-12)


class GlottalSource:
    def __init__(self, fs: float, shape: str = "rosenberg"):
        self.fs = fs
        self.shape = shape
        self._phase = 0.0  # 归一化相位 [0, 1)
        self.rng = np.random.default_rng(0)
        self.tilt = SpectralTilt(fs)

    def reset(self) -> None:
        self._phase = 0.0
        self.tilt.reset()

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
                       noise_gain: float | np.ndarray = 0.0,
                       voiced_gain: float | np.ndarray = 1.0,
                       tilt_db: float | np.ndarray = 0.0) -> np.ndarray:
        """生成 n 个激励样本。

        f0 / noise_gain / voiced_gain / tilt_db 均可为标量或长度 n 的数组。
        tilt_db 只作用于有声部分，气声噪声在倾斜层之后独立加入。
        """
        f0_arr = np.full(n, float(f0)) if np.isscalar(f0) else np.asarray(f0, float)
        phases = self._advance_phase(n, f0_arr)
        voiced = self._periodic(phases, open_quotient, f0_arr)

        tilt = np.broadcast_to(np.asarray(tilt_db, float), (n,))
        if np.any(tilt != 0.0):
            # m 随样本变化时需分段处理，块内保持低通状态连续
            out = np.empty(n)
            start = 0
            while start < n:
                stop = start + 1
                while stop < n and tilt[stop] == tilt[start]:
                    stop += 1
                mix = self.tilt.mix_for_db(tilt[start])
                out[start:stop] = self.tilt.process(voiced[start:stop], mix)
                start = stop
            voiced = out
        else:
            self.tilt.reset()

        noise = self.rng.standard_normal(n)
        gain = np.broadcast_to(np.asarray(voiced_gain, float), (n,))
        ngain = np.broadcast_to(np.asarray(noise_gain, float), (n,))
        return gain * voiced + ngain * noise
