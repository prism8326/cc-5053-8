"""引擎与端到端参数关系测试。"""

import os
import tempfile

import numpy as np

from vowel_synth.analysis import (estimate_formant_peaks,
                                  estimate_harmonic_spacing,
                                  verify_formants)
from vowel_synth.audio_io import read_wav, write_wav
from vowel_synth.engine import RealtimeEngine, SynthParams, render
from vowel_synth.vowels import get_vowel


def _vowel_params(f0=120.0, symbol="a"):
    f, b = get_vowel(symbol)
    return SynthParams(f0=f0, freqs=f, bws=b, noise_gain=0.0)


def test_render_shapes_and_finite():
    p = _vowel_params()
    out = render(0.5, 22050, p)
    n = int(0.5 * 22050)
    assert out["y"].shape == (n,)
    assert out["source"].shape == (n,)
    assert np.all(np.isfinite(out["y"]))
    assert np.max(np.abs(out["y"])) <= 0.9 + 1e-9  # 归一化


def test_pitch_does_not_move_formants():
    """核心解耦：f0 改变时，实际共振峰系数必须逐块完全相同。"""
    p1 = _vowel_params(f0=100.0)
    p2 = _vowel_params(f0=220.0)
    r1 = render(0.4, 22050, p1, block_size=512)
    r2 = render(0.4, 22050, p2, block_size=512)
    np.testing.assert_allclose(r1["effective"][:, :, 0],
                               r2["effective"][:, :, 0], atol=1e-10)
    np.testing.assert_allclose(r1["effective"][:, :, 1],
                               r2["effective"][:, :, 1], atol=1e-10)


def test_spectrum_harmonic_spacing_tracks_f0():
    for f0 in (90.0, 150.0, 240.0):
        out = render(0.8, 22050, _vowel_params(f0), block_size=512)
        est = estimate_harmonic_spacing(out["y"], 22050)
        assert abs(est - f0) < 4.0, f"f0={f0} 估计={est}"


def test_formant_envelope_peaks_near_targets():
    """白噪激励实测传递函数，分离良好的共振峰应在设计频率 ±5% 内。

    F3/F4 间距小于约一个峰宽时会物理合并，属正常现象，
    因此这里只核验间距充足的 F1-F3（选 a/ɔ/ə 等元音）。
    """
    for symbol in ("a", "ɔ", "ə", "ɛ"):
        p = _vowel_params(symbol=symbol)
        results = verify_formants(p.freqs[:3], p.bws[:3], 22050)
        for r in results:
            assert r["ok"], (f"{symbol} 目标 {r['target']:.0f}Hz "
                             f"实测 {r['measured']:.0f}Hz "
                             f"相对误差 {r['rel_error']:.3f}")


def test_measured_transfer_matches_designed_response():
    """实测 H(f) 与设计系数频率响应的峰位应一致（直接对照滤波器理论）。"""
    from vowel_synth.filters import FormantCascade
    from vowel_synth.analysis import measure_transfer_function
    p = _vowel_params(symbol="a")
    f_meas, h_meas = measure_transfer_function(p.freqs, p.bws, 22050)
    cas = FormantCascade(p.n_formants, 22050)
    f_des, h_des = cas.frequency_response(p.freqs, p.bws, n_fft=8192)
    band = (f_des >= 100) & (f_des <= 5000)
    # 两条曲线形状相关（dB 域），证明实测与理论响应同源
    h_meas_resamp = np.interp(f_des[band], f_meas, h_meas)
    corr = np.corrcoef(h_meas_resamp, h_des[band])[0, 1]
    assert corr > 0.9


def test_voiced_lpc_rough_formant_estimate():
    """有声信号的 LPC 粗估计至少能在 F1/F2 邻域给出峰（宽松教学检查）。"""
    p = _vowel_params(symbol="a")
    out = render(1.5, 22050, p, block_size=512)
    peaks = estimate_formant_peaks(out["y"], 22050, n_peaks=4)
    assert peaks.size >= 2
    for target in p.freqs[:2]:
        assert np.min(np.abs(peaks - target)) < 0.25 * target


def test_pitch_sweep_continuous_and_formants_constant():
    p = _vowel_params()
    duration, block_size = 0.75, 512
    n = int(round(duration * 22050))
    n_blocks = int(np.ceil(n / block_size))
    f0_env = np.linspace(80, 400, n_blocks)
    out = render(duration, 22050, p, block_size=block_size, f0_envelope=f0_env)
    # 输出连续：以典型相邻差（95 分位）为基准，最大跳变不得异常放大
    # （相位累加器逐样本推进，块边界不应出现脉冲式跳变）
    d = np.abs(np.diff(out["y"]))
    typical = np.percentile(d, 95)
    assert d.max() < 8.0 * typical + 1e-6
    # 生效共振峰频率在整个扫描中零漂移
    drift = np.abs(out["effective"][:, :, 0] -
                   out["effective"][0, :, 0]).max()
    assert drift < 1e-10
    # f0 轨迹确实在扫描
    assert out["f0_track"][0] < out["f0_track"][-1]


def test_out_of_range_params_record_warnings_but_render():
    f, b = get_vowel("a")
    f_bad = f.copy()
    f_bad[0] = 20000.0          # 越过 Nyquist
    b_bad = b.copy()
    b_bad[1] = 1.0              # 过窄
    p = SynthParams(f0=120.0, freqs=f_bad, bws=b_bad)
    out = render(0.3, 22050, p, block_size=512)
    assert out["warnings"], "越界参数必须产生显式告警"
    assert np.all(np.isfinite(out["y"]))


def test_realtime_engine_block_streaming():
    p = _vowel_params()
    eng = RealtimeEngine(22050, p, block_size=256)
    blocks = [eng.process_block() for _ in range(8)]
    y = np.concatenate(blocks)
    assert y.shape == (2048,) and np.all(np.isfinite(y))
    # 运行中改 f0 不改变共振峰系数
    p2 = SynthParams(f0=300.0, freqs=p.freqs.copy(), bws=p.bws.copy())
    eng.update_params(p2)
    eng.process_block()
    freqs_after = np.array([s.f_effective for s in eng.last_specs])
    np.testing.assert_allclose(freqs_after, p.freqs, atol=1e-9)


def test_wav_roundtrip():
    p = _vowel_params()
    out = render(0.3, 22050, p)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.wav")
        write_wav(path, out["y"], 22050)
        y2, fs2 = read_wav(path)
        assert fs2 == 22050
        assert y2.shape == out["y"].shape
        assert np.corrcoef(y2, out["y"])[0, 1] > 0.999
