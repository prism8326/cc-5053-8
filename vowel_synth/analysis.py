"""频谱检查工具：验证谐波间距与共振包络的参数关系。

两类测量对应两种物理量，不能混用：
- 谐波间距（有声输出的谱线间隔）→ 估计基频 f0；
- 共振峰（声道传递函数的极点）→ 用白噪激励测量 H(f) 得到，
  因为白噪谱平坦，不会被声门源的谱倾斜和谐波梳齿污染。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, welch, windows

from .filters import FormantCascade


def magnitude_spectrum(x: np.ndarray, fs: float, n_fft: int | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """加 Hann 窗的单边幅度谱 (频率 Hz, 线性幅度)。"""
    x = np.asarray(x, float)
    n_fft = n_fft or int(2 ** np.ceil(np.log2(max(1024, x.size))))
    w = windows.hann(x.size, sym=False)
    X = np.fft.rfft(x * w, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    return freqs, np.abs(X) / (0.5 * w.sum())


def estimate_harmonic_spacing(x: np.ndarray, fs: float,
                              f_min: float = 60.0,
                              f_max: float = 500.0) -> float:
    """从频谱峰间距估计基频（谐波间距 = f0）。"""
    freqs, mag = magnitude_spectrum(x, fs)
    band = (freqs >= f_min) & (freqs <= f_max)
    idx, _ = find_peaks(mag[band], height=mag.max() * 0.05)
    peak_freqs = freqs[band][idx]
    if peak_freqs.size < 2:
        return float("nan")
    return float(np.median(np.diff(peak_freqs)))


def harmonic_peaks(x: np.ndarray, fs: float, n_harmonics: int = 12
                   ) -> tuple[np.ndarray, np.ndarray]:
    """返回前若干条谐波的 (频率, 幅度)，用于直观核对等间距。"""
    freqs, mag = magnitude_spectrum(x, fs)
    f0 = estimate_harmonic_spacing(x, fs)
    if not np.isfinite(f0):
        return np.array([]), np.array([])
    out_f, out_m = [], []
    for h in range(1, n_harmonics + 1):
        target = h * f0
        if target >= fs / 2:
            break
        j = int(np.argmin(np.abs(freqs - target)))
        out_f.append(freqs[j])
        out_m.append(mag[j])
    return np.array(out_f), np.array(out_m)


def measure_transfer_function(freqs: np.ndarray, bws: np.ndarray,
                              fs: float, duration: float = 4.0,
                              seed: int = 7
                              ) -> tuple[np.ndarray, np.ndarray]:
    """用平坦白噪激励共振峰级联，实测传递函数 (频率 Hz, 幅度 dB)。

    这是验证共振峰参数最直接的方法：输入谱无倾斜、无谐波栅格，
    输出谱即系统 H(f) 的估计。
    """
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    x = rng.standard_normal(n)
    cascade = FormantCascade(len(freqs), fs)
    y, _ = cascade.process_block(x, np.asarray(freqs, float),
                                 np.asarray(bws, float))
    f, Pxx = welch(x, fs, nperseg=4096, nfft=8192)
    _, Pyy = welch(y, fs, nperseg=4096, nfft=8192)
    H_db = 10.0 * np.log10((Pyy / (Pxx + 1e-15)) + 1e-15)
    return f, H_db


def estimate_formant_peaks(x: np.ndarray, fs: float,
                           n_peaks: int = 4,
                           f_max: float | None = None
                           ) -> np.ndarray:
    """从有声信号的 LPC 全极点包络估计共振峰（粗估计，供快速自检）。

    精确验证请用 measure_transfer_function 的白噪法；本函数对靠近的
    共振峰（如 F3/F4）可能合并，结果按频率升序返回。
    """
    x = np.asarray(x, float)
    order = min(int(fs / 1000.0) + 4, x.size // 2 - 2)
    order = max(order, 8)
    xw = x * windows.hann(x.size, sym=False)
    r = np.correlate(xw, xw, mode="full")[x.size - 1:][:order + 1]
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = max(r[0], 1e-12)
    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1])
        k = float(np.clip(-acc / e, -0.999, 0.999))
        prev = a[:i].copy()
        a[i] = k
        a[1:i] = prev[1:i] + k * prev[i - 1:0:-1]
        e *= 1.0 - k ** 2
    n_fft = 4096
    A = np.fft.rfft(a, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    env_db = 20.0 * np.log10(1.0 / (np.abs(A) + 1e-12))
    f_max = f_max or 0.45 * fs
    band = (freqs >= 120.0) & (freqs <= f_max)
    idx, props = find_peaks(env_db[band], prominence=2.0)
    peak_f, peak_p = freqs[band][idx], props["prominences"]
    keep = np.argsort(peak_p)[::-1][:n_peaks]
    return np.sort(peak_f[keep])


def verify_formants(freqs: np.ndarray, bws: np.ndarray, fs: float,
                    tolerance: float = 0.05
                    ) -> list[dict]:
    """白噪法逐峰核验：返回每个设计频率的实测峰位与相对误差。"""
    f, H_db = measure_transfer_function(freqs, bws, fs)
    results = []
    for target in np.asarray(freqs, float):
        lo, hi = target * 0.85, target * 1.15
        m = (f >= lo) & (f <= hi)
        if not m.any():
            results.append({"target": target, "measured": float("nan"),
                            "rel_error": float("nan"), "ok": False})
            continue
        sub_f, sub_h = f[m], H_db[m]
        idx, _ = find_peaks(sub_h)
        measured = sub_f[idx[np.argmax(sub_h[idx])]] if idx.size else \
            sub_f[np.argmax(sub_h)]
        rel = abs(measured - target) / target
        results.append({"target": float(target), "measured": float(measured),
                        "rel_error": float(rel), "ok": bool(rel <= tolerance)})
    return results
