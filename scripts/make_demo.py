"""离线验证脚本：生成可听 WAV 与一张综合频谱关系图。

运行：.venv/bin/python scripts/make_demo.py
产物全部写入 demo_output/。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vowel_synth.analysis import (estimate_harmonic_spacing,
                                  magnitude_spectrum,
                                  measure_transfer_function,
                                  verify_formants)
from vowel_synth.audio_io import write_wav
from vowel_synth.engine import SynthParams, render
from vowel_synth.vowels import get_vowel, list_ascii, list_symbols

FS = 22050
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "demo_output")
os.makedirs(OUT, exist_ok=True)


def main() -> None:
    # 1) 每个元音各渲染一个 WAV
    for sym, ascii_name in zip(list_symbols(), list_ascii()):
        f, b = get_vowel(sym)
        p = SynthParams(f0=120.0, freqs=f, bws=b, noise_gain=0.02)
        r = render(1.0, FS, p)
        write_wav(f"{OUT}/vowel_{ascii_name}.wav", r["y"], FS)

    # 2) 同一元音两个音高：证明谐波间距变、共振包络不变
    f, b = get_vowel("a")
    r_lo = render(1.0, FS, SynthParams(f0=90.0, freqs=f, bws=b,
                                       noise_gain=0.0))
    r_hi = render(1.0, FS, SynthParams(f0=240.0, freqs=f, bws=b,
                                       noise_gain=0.0))
    write_wav(f"{OUT}/a_f090.wav", r_lo["y"], FS)
    write_wav(f"{OUT}/a_f0240.wav", r_hi["y"], FS)
    print("f0=90  谐波间距估计:",
          round(estimate_harmonic_spacing(r_lo["y"], FS), 2), "Hz")
    print("f0=240 谐波间距估计:",
          round(estimate_harmonic_spacing(r_hi["y"], FS), 2), "Hz")
    # 两次渲染的实际共振峰系数必须完全相同
    np.testing.assert_allclose(r_lo["effective"], r_hi["effective"],
                               atol=1e-12)
    print("两音高下共振峰系数最大差异:",
          float(np.abs(r_lo["effective"] - r_hi["effective"]).max()))

    # 3) 连续音高扫描
    n = int(1.5 * FS)
    n_blocks = int(np.ceil(n / 512))
    sweep = render(1.5, FS, SynthParams(f0=90.0, freqs=f, bws=b),
                   block_size=512, f0_envelope=np.linspace(80, 400, n_blocks))
    write_wav(f"{OUT}/pitch_sweep.wav", sweep["y"], FS)
    print("扫描期共振峰漂移:",
          float(np.abs(sweep["effective"][:, :, 0]
                       - sweep["effective"][0, :, 0]).max()), "Hz")

    # 4) 越界与过窄带宽：钳制而非崩溃
    bad = SynthParams(f0=120.0,
                      freqs=np.array([20000., 1090., 2440., 3500.]),
                      bws=np.array([80., 1., 120., 180.]))
    r_bad = render(0.5, FS, bad)
    write_wav(f"{OUT}/clamped_params.wav", r_bad["y"], FS)
    print("越界渲染告警数:", len(r_bad["warnings"]))
    for w in r_bad["warnings"][0]["messages"]:
        print("  -", w)

    # 5) 综合验证图
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for r, name, c in [(r_lo, "f0=90 Hz", "tab:blue"),
                       (r_hi, "f0=240 Hz", "tab:red")]:
        fr, mag = magnitude_spectrum(r["y"], FS)
        m = fr <= 4000
        axes[0, 0].plot(fr[m], 20 * np.log10(mag[m] + 1e-9),
                        label=name, lw=0.8, color=c)
    for ff in f:
        axes[0, 0].axvline(ff, color="green", ls="--", lw=0.7)
    axes[0, 0].set_title("Same vowel, two pitches: harmonic spacing moves, "
                         "formants (green) fixed")
    axes[0, 0].set_xlabel("frequency (Hz)")
    axes[0, 0].set_ylabel("dB")
    axes[0, 0].legend(fontsize=8)

    # 扫描的频谱图
    nfft = 1024
    hop = 256
    frames = []
    for i in range(0, len(sweep["y"]) - nfft, hop):
        seg = sweep["y"][i:i + nfft] * np.hanning(nfft)
        frames.append(np.abs(np.fft.rfft(seg, n=nfft)))
    S = np.array(frames).T
    fax = np.fft.rfftfreq(nfft, 1 / FS)
    tax = np.arange(S.shape[1]) * hop / FS
    axes[0, 1].pcolormesh(tax, fax[:250], 20 * np.log10(S[:250] + 1e-9),
                          shading="auto", cmap="magma")
    for ff in f[:3]:
        axes[0, 1].axhline(ff, color="lime", ls="--", lw=0.8)
    axes[0, 1].set_title("Pitch sweep spectrogram: harmonics bend upward, "
                         "formant lines stay level")
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("frequency (Hz)")

    # 实测传递函数 vs 设计频率
    fm, Hm = measure_transfer_function(f, b, FS)
    axes[1, 0].plot(fm[fm <= 5000], Hm[fm <= 5000], label="measured H(f)")
    for ff in f:
        axes[1, 0].axvline(ff, color="green", ls="--", lw=0.7)
    axes[1, 0].set_title("White-noise measured transfer function vs targets")
    axes[1, 0].set_xlabel("frequency (Hz)")
    axes[1, 0].set_ylabel("dB")

    # 带宽对峰尖锐度的影响
    for bw_scale, c in [(0.5, "tab:blue"), (1.0, "tab:orange"),
                        (2.0, "tab:red")]:
        fm2, H2 = measure_transfer_function(
            f, b * bw_scale, FS, duration=2.0)
        axes[1, 1].plot(fm2[fm2 <= 1500], H2[fm2 <= 1500],
                        label=f"bw x{bw_scale}", color=c)
    axes[1, 1].set_title("Bandwidth controls peak sharpness (F1 region)")
    axes[1, 1].set_xlabel("frequency (Hz)")
    axes[1, 1].set_ylabel("dB")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT}/verification.png", dpi=110)
    print("白噪法共振峰核验:")
    for r in verify_formants(f[:3], b[:3], FS):
        print(f"  target {r['target']:7.1f} -> measured {r['measured']:7.1f}"
              f" ({r['rel_error']*100:4.2f}%)")
    print("产物已写入", OUT)


if __name__ == "__main__":
    main()
