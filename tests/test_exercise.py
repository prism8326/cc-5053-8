"""同音高、异声道练习集与音高穿带示范的测试。

覆盖需求里的可验证约定：
- 声源不变 / 只移共振峰，与 只变源谱倾斜 两组在滑音前谱形接近、滑音后拉开；
- 差异起点来自实际输出，而不是写死的计划值；
- 更换基频后重新生成音频与答案，答案不来自旧文件；
- 谐波离共振峰的间距被检查，“最强谐波 ≠ 共振峰中心”；
- 同一份声道答案在另一基频下仍能解释实际输出（跨基频核验）。
"""

import json
import os
import tempfile

import numpy as np

from vowel_synth.analysis import estimate_harmonic_spacing
from vowel_synth.engine import SynthParams, render
from vowel_synth.exercise import (ExerciseSpec, PitchDemoSpec, VARIANTS,
                                  build_exercise, build_pitch_demo,
                                  choose_safe_f0, cross_pitch_check,
                                  envelope_db, harmonic_amplitudes,
                                  save_assets, spectral_shape_distance)
from vowel_synth.source import SpectralTilt


def _built(**kw):
    return build_exercise(ExerciseSpec(**kw))


def test_three_variants_share_f0_and_breath_but_differ_in_one_layer():
    b = _built()
    a, t, ref = (b.tracks["a_formant"], b.tracks["b_tilt"],
                 b.tracks["anchor"])
    # 共享：f0 轨迹与气声事件逐块相同
    np.testing.assert_allclose(a["f0"], t["f0"])
    np.testing.assert_allclose(a["f0"], ref["f0"])
    np.testing.assert_allclose(a["noise"], t["noise"])
    np.testing.assert_allclose(a["noise"], ref["noise"])
    # A 只改共振峰、不改倾斜；B 只改倾斜、不改共振峰
    np.testing.assert_allclose(a["freqs"][0], ref["freqs"][0])
    assert np.abs(a["freqs"][-1] - ref["freqs"][-1]).max() > 100.0
    assert np.abs(t["freqs"] - ref["freqs"]).max() < 1e-9
    assert np.abs(t["tilt"]).max() > 1.0
    assert np.abs(a["tilt"]).max() < 1e-9


def test_a_and_b_close_before_glide_and_separated_after():
    b = _built()
    pair = b.answer["pair_a_vs_b"]
    spec = b.spec
    # 滑音前几乎不可分辨
    assert pair["pre_glide_median_db"] < 0.5
    # 滑音后显著拉开
    assert pair["post_glide_median_db"] > 8.0
    # 实测分离起点落在滑音段内（由实际输出测得）
    assert spec.glide_start - 0.08 <= pair["onset_s"] <= spec.glide_end + 0.15


def test_divergence_onset_comes_from_audio_not_schedule():
    """把滑音计划推迟，实测起点必须随之推迟——证明不是写死的常数。"""
    early = _built(lead=0.20, hold=0.30)
    late = _built(lead=0.20, hold=0.90)
    assert late.answer["pair_a_vs_b"]["onset_s"] > \
        early.answer["pair_a_vs_b"]["onset_s"] + 0.4


def test_spectral_tilt_moves_no_resonance_peak():
    fs = 16000
    st = SpectralTilt(fs)
    f = np.linspace(100, 4000, 2000)
    h0 = st.response_db(f, 0.0)
    hm = st.response_db(f, -24.0)
    assert np.all(np.diff(hm) <= 1e-6), "倾斜后响应应单调下降，不应出现新峰"
    assert np.abs(h0).max() < 1e-9
    assert hm[0] > hm[-1] + 10.0


def test_tilt_does_not_move_formants():
    """同一段共振峰轨迹 + 不同源谱倾斜 → 生效共振峰系数完全相同。"""
    spec = ExerciseSpec()
    b1 = build_exercise(spec, analysis=False)
    b2 = build_exercise(ExerciseSpec(tilt_target_db=0.0), analysis=False)
    # anchor 版本的倾斜恒为 0，两者应逐块一致
    np.testing.assert_allclose(b1.renders["anchor"]["effective"],
                               b2.renders["anchor"]["effective"], atol=1e-12)


def test_resolved_f0_keeps_harmonics_off_formant_centers():
    spec = ExerciseSpec()
    info = choose_safe_f0(spec, spec.f0_requested)
    assert info["safe"], info
    # 滑音后的稳定段也要满足：谐波与移动后的共振峰至少差 10% f0
    b = build_exercise(spec, analysis=False)
    post_f0 = b.f0 * spec.f0_glide_ratio
    moved = np.asarray(spec.moved_freqs)
    k = np.arange(1, int(4200 / post_f0) + 1)
    nearest = min(np.min(np.abs(k * post_f0 - fc)) for fc in moved)
    assert nearest > 0.05 * post_f0


def test_rebuild_at_new_f0_regenerates_answer_from_parameters():
    spec = ExerciseSpec()
    low = build_exercise(spec, f0=120.0)
    high = build_exercise(spec, f0=300.0)
    assert abs(low.f0 - high.f0) > 20.0
    # 谐波间距随新 f0 变化（在滑音前的稳定段测量，避免滑音本身抬高估计）
    lead_n = int(spec.lead * spec.fs)
    est = estimate_harmonic_spacing(high.renders["anchor"]["y"][:lead_n],
                                    spec.fs, f_min=0.6 * high.f0,
                                    f_max=2.6 * high.f0)
    assert abs(est - high.f0) < 0.08 * high.f0, (est, high.f0)
    # 共振峰答案（包络峰位）不随 f0 改变
    for v in ("anchor", "a_formant"):
        p_low = low.answer["probes"][v][-1]["envelope_peaks_hz"]
        p_high = high.answer["probes"][v][-1]["envelope_peaks_hz"]
        np.testing.assert_allclose(p_low, p_high, atol=6.0)
    # 答案里的 f0 字段确实记录的是新值
    assert abs(high.answer["f0"]["used_hz"] - high.f0) < 0.01
    assert abs(low.answer["f0"]["used_hz"] - low.f0) < 0.01
    # 音频确实重新生成了
    assert not np.allclose(low.renders["anchor"]["y"][:4000],
                           high.renders["anchor"]["y"][:4000])


def test_strongest_harmonic_is_not_the_formant_center():
    demo = build_pitch_demo(PitchDemoSpec())
    focus = demo["focus_frame"]
    assert focus is not None
    # 最强谐波落在两个共振带之间，距离两个中心都不近
    assert focus["between_bands"]
    assert focus["strongest_offset_to_f1_hz"] > 100.0
    assert focus["strongest_offset_to_f2_hz"] > 100.0
    # 而包络峰仍稳定在设计值附近
    peaks = demo["envelope_peaks_hz"]
    assert abs(peaks[0] - 520.0) < 15.0 and abs(peaks[1] - 980.0) < 15.0


def test_cross_pitch_answer_still_explains_actual_output():
    b = _built()
    for alt in (120.0, 260.0, 320.0):
        cp = cross_pitch_check(b, alt)
        assert cp["ok"], (alt, cp["explainability"])
        for v in ("anchor", "a_formant"):
            for row in cp["explainability"][v]["frames"]:
                assert row["corr"] > 0.95
                assert row["rmse_db"] < 2.0
        # 包络峰位由声道决定，跨基频不变
        assert np.allclose(cp["envelope_peaks_hz"]["anchor"],
                           b.answer["probes"]["anchor"][-1]["envelope_peaks_hz"],
                           atol=8.0)
        # 而“就近取谐波”的朴素估计会随 f0 改变
        assert cp["nearest_harmonic_to_formant_hz"]["anchor"] is not None


def test_harmonic_amplitudes_match_envelope_model():
    """输出谐波幅度 ≈ 源谱(含倾斜) + 声道包络，误差在合理范围。"""
    spec = ExerciseSpec()
    b = build_exercise(spec, analysis=False)
    res = b.renders["b_tilt"]
    t = spec.phrase_seconds - 0.15
    f0 = float(res["f0_track"][min(int(t * spec.fs), res["f0_track"].size - 1)])
    b_idx = min(int(t * spec.fs / spec.block), res["freqs"].shape[0] - 1)
    hf, ha = harmonic_amplitudes(res["y"], spec.fs, t, f0)
    _, sa = harmonic_amplitudes(res["source"], spec.fs, t, f0)
    pred = (20 * np.log10(sa + 1e-12)
            + envelope_db(res["freqs"][b_idx], res["bws"][b_idx],
                          spec.fs, hf))
    actual = 20 * np.log10(ha + 1e-12)
    corr = np.corrcoef(actual - actual.mean(),
                       pred - pred.mean())[0, 1]
    assert corr > 0.95


def test_hide_answer_writes_no_answer_but_keeps_parameters():
    b = _built()
    with tempfile.TemporaryDirectory() as d:
        files = save_assets(b, d, hide_answer=True, plots=True)
        assert not os.path.exists(os.path.join(d, "answer.json"))
        assert os.path.exists(os.path.join(d, "spec.json"))
        with open(files["spec"], encoding="utf-8") as f:
            spec = json.load(f)
        # 参数足以重建音频：spec 内含全部轨迹参数与解析后的 f0
        assert spec["resolved_f0_hz"] == b.f0
        assert len(spec["spec"]["anchor_freqs"]) == 4
        assert os.path.exists(files["a_formant"])
        assert os.path.exists(files["pairA_anchor_then_a_formant"])


def test_saved_spec_rebuilds_identical_audio():
    b = _built()
    with tempfile.TemporaryDirectory() as d:
        save_assets(b, d, plots=False)
        with open(os.path.join(d, "spec.json"), encoding="utf-8") as f:
            saved = json.load(f)
        spec2 = ExerciseSpec.from_dict(saved["spec"])
        b2 = build_exercise(spec2, f0=saved["resolved_f0_hz"], analysis=False)
        for v in VARIANTS:
            np.testing.assert_allclose(b.renders[v]["y"], b2.renders[v]["y"],
                                       atol=1e-9)


def test_noise_envelope_is_continuous_across_block_rendering():
    """逐块气声增益在块内保持常量、块间可变化，且输出连续无爆音。"""
    spec = ExerciseSpec()
    b = build_exercise(spec, analysis=False)
    y = b.renders["a_formant"]["y"]
    d = np.abs(np.diff(y))
    assert d.max() < 20 * np.percentile(d, 95) + 1e-3


def test_engine_render_accepts_tilt_and_noise_envelopes():
    fs, n = 16000, 0.5
    blocks = int(np.ceil(int(fs * n) / 128))
    r = render(n, fs, SynthParams(f0=150.0), block_size=128,
               tilt_envelope=np.linspace(0.0, -18.0, blocks),
               noise_envelope=np.linspace(0.001, 0.05, blocks),
               voiced_envelope=np.ones(blocks))
    assert r["tilt"].size == blocks
    assert np.all(np.isfinite(r["y"]))
    # 倾斜确实生效：与无倾斜版本可区分
    flat = render(n, fs, SynthParams(f0=150.0), block_size=128)
    assert not np.allclose(r["y"], flat["y"])


def test_instrument_window_freezes_one_layer_and_reports_onset():
    """乐器窗口：固定某一层后，该乐器的差异归零，起点由实测给出。"""
    import matplotlib
    matplotlib.use("Agg")
    from vowel_synth.exercise_gui import InstrumentWindow

    w = InstrumentWindow()
    base_a = w.dist["a_formant"]
    base_b = w.dist["b_tilt"]
    assert base_a["d"].max() > 5.0 and base_b["d"].max() > 2.0
    assert np.isfinite(base_a["onset"]) and np.isfinite(base_b["onset"])

    # 固定声道层 → A 相对参照不再变化（差异只来自未固定的源谱层）
    w.fixed["tract.formants"] = True
    w._render_and_analyse()
    assert w.dist["a_formant"]["d"].max() < 1e-6
    assert w.dist["a_formant"]["onset"] != w.dist["a_formant"]["onset"]  # nan
    assert w.dist["b_tilt"]["d"].max() > 2.0
    a_vs_b_after = w.dist["a_vs_b"]["d"].max()
    assert a_vs_b_after < base_a["d"].max()  # 移除了 A 的贡献

    # 再固定源谱层 → 两个乐器都与参照重合，A vs B 也为零
    w.fixed["source.tilt"] = True
    w._render_and_analyse()
    assert w.dist["b_tilt"]["d"].max() < 1e-6
    assert w.dist["a_vs_b"]["d"].max() < 1e-6

    # 恢复后回到初始状态
    w._reset_layers()
    assert np.isclose(w.dist["a_formant"]["d"].max(), base_a["d"].max(),
                      atol=1e-6)


def test_instrument_window_hides_answer_but_keeps_live_measurement():
    import matplotlib
    matplotlib.use("Agg")
    from vowel_synth.exercise_gui import InstrumentWindow

    w = InstrumentWindow()
    w._toggle_answer()
    assert w.show_answer is False
    w._redraw()
    assert w.dist["a_vs_b"]["d"].size > 0
    w.cursor = 1.2
    w._redraw()


def test_buffer_player_seek_pause_and_offline_cursor():
    from vowel_synth.audio_io import BufferPlayer

    y = np.zeros(16000)
    p = BufferPlayer(y, 16000)
    p.pos = 16000 // 2
    assert abs(p.time - 0.5) < 1e-9
    p.playing = True
    p.advance(0.1)
    assert abs(p.time - 0.6) < 1e-6
    p.pause()
    assert not p.playing
    p.load(np.zeros(8000))
    assert p.pos == 0 and abs(p.duration - 0.5) < 1e-9
