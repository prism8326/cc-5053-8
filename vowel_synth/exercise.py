"""同音高、异声道的元音练习集：参数化谱面 → 可重建音频 + 答案轨迹。

练习的结构
----------
每个练习由三层的**参数轨迹**描述，而不是预录音频或预录答案：

1. 声源层：f0 曲线、声门开相、源谱倾斜（tilt_db）。
2. 声道层：F1..F4 中心频率与带宽的逐块轨迹。
3. 气声层：送气噪声包络与显式气声事件。

练习包含一个共享参照 ``anchor`` 和两个目标：

- ``a_formant``：声源（f0、倾斜、气声）逐样本不变，只有共振峰在滑音段移动；
- ``b_tilt``：声道逐样本不变，只有源谱倾斜在滑音段改变。

所有版本共享同一段 f0 滑音与同一气声事件，因此滑音之前三个版本的谱形
几乎逐采样相同；“只有滑音后才容易分辨”。答案轨迹由参数直接计算，
更换基频会用同一套参数重新合成并重新分析，不读取任何旧答案。

谐波与共振峰的区分
------------------
答案里的共振峰来自声道传递函数的极点（与 f0 无关）；输出谱里的强线
来自谐波（间距 = f0）。生成时会搜索一个“安全”的 f0，使锚点与移动后的
共振峰中心都不落在任何谐波附近，并把最坏偏移写进答案。跨基频核验再用
另一 f0 重新生成音频，用实测谐波幅度检验同一份声道包络是否仍能解释输出。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import find_peaks, windows

from .audio_io import write_wav
from .engine import SynthParams, render
from .filters import FormantCascade

FS_DEFAULT = 16000          # 16 kHz：F1..F4 与 4 kHz 内的判别都够用
BLOCK_DEFAULT = 256         # 16 ms 一块，62.5 帧/秒
BAND_HZ = (150.0, 4200.0)   # 谱形比较与谐波分析的有效频带
MAX_ANALYSIS_HZ = 4200.0


# --------------------------------------------------------------------------
# 谱面参数
# --------------------------------------------------------------------------
@dataclass
class ExerciseSpec:
    """可编辑的练习谱面。所有时间单位为秒。"""

    name: str = "homophone-multichannel"
    fs: int = FS_DEFAULT
    block: int = BLOCK_DEFAULT
    seed: int = 20261120

    f0_requested: float = 176.0
    f0_glide_ratio: float = 1.25        # 滑音后 f0 相对滑音前的倍数
    anchor_freqs: tuple = (500.0, 1400.0, 2500.0, 3400.0)
    anchor_bws: tuple = (70.0, 90.0, 130.0, 180.0)
    moved_freqs: tuple = (820.0, 1950.0, 2700.0, 3400.0)
    tilt_target_db: float = -24.0       # 目标版本的源谱倾斜（高频相对低频）
    # 滑音前两组各自的“基线偏移”占完整变化量的比例。默认 0：滑音前两组
    # 谱形完全一致（“只有滑音后才容易分辨”）；调大它可增加听辨难度。
    baseline_frac: float = 0.0

    open_quotient: float = 0.55
    shape: str = "rosenberg"

    lead: float = 0.30                  # 滑音前的稳定段
    hold: float = 0.55
    glide: float = 0.55                 # f0 与“变化层”同时滑动的区间
    hold2: float = 0.55

    noise_floor: float = 0.006
    breath_peak: float = 0.075
    breath_delay: float = 0.10          # 气声事件相对滑音起点的延迟
    breath_width: float = 0.14          # 半宽（升余弦）

    clearance_target: float = 0.15      # 谐波-共振峰最小间距 / f0 的目标比值

    def __post_init__(self) -> None:
        self.anchor_freqs = tuple(float(v) for v in self.anchor_freqs)
        self.anchor_bws = tuple(float(v) for v in self.anchor_bws)
        self.moved_freqs = tuple(float(v) for v in self.moved_freqs)
        if len(self.anchor_freqs) != 4 or len(self.anchor_bws) != 4:
            raise ValueError("本练习集固定使用 4 个共振峰")
        if len(self.moved_freqs) != 4:
            raise ValueError("moved_freqs 必须提供 4 个中心频率")

    @property
    def phrase_seconds(self) -> float:
        return self.lead + self.hold + self.glide + self.hold2

    @property
    def glide_start(self) -> float:
        return self.lead + self.hold

    @property
    def glide_end(self) -> float:
        return self.lead + self.hold + self.glide

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExerciseSpec":
        return cls(**d)


# --------------------------------------------------------------------------
# 时间与轨迹
# --------------------------------------------------------------------------
def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def block_times(spec: ExerciseSpec) -> np.ndarray:
    n = int(round(spec.phrase_seconds * spec.fs / spec.block))
    return (np.arange(n) + 0.5) * spec.block / spec.fs


def _glide_alpha(t: np.ndarray, spec: ExerciseSpec) -> np.ndarray:
    return smoothstep((t - spec.glide_start) / spec.glide)


def f0_track(spec: ExerciseSpec, f0: float) -> np.ndarray:
    alpha = _glide_alpha(block_times(spec), spec)
    return f0 * (1.0 + (spec.f0_glide_ratio - 1.0) * alpha)


def breath_envelope(spec: ExerciseSpec) -> np.ndarray:
    t = block_times(spec)
    center = spec.glide_start + spec.breath_delay
    # 升余弦气声事件；两端为 0，因此不会在谱面外截断
    u = np.abs(t - center) / spec.breath_width
    bump = np.where(u < 1.0, 0.5 * (1.0 + np.cos(np.pi * np.clip(u, 0, 1))), 0.0)
    return spec.noise_floor + spec.breath_peak * bump


def layer_tracks(spec: ExerciseSpec, variant: str, f0: float) -> dict:
    """单个版本的逐块三层轨迹（未渲染）。

    A/B 两组都从同一参照出发，但各自带一个极小的基线偏移（baseline_frac），
    滑音前接近而不相同；滑音段 A 把偏移渐变为“移动后的共振峰”，
    B 把偏移渐变为“改变后的源谱倾斜”。
    """
    t = block_times(spec)
    alpha = _glide_alpha(t, spec)
    base = spec.baseline_frac
    anchor = np.asarray(spec.anchor_freqs, float)
    freqs = np.tile(anchor, (t.size, 1))
    tilt = np.zeros(t.size)
    if variant == "a_formant":
        target = np.asarray(spec.moved_freqs, float)
        base_f = anchor + base * (target - anchor)
        freqs = (np.tile(base_f, (t.size, 1))
                 + alpha[:, None] * (target - base_f))
    elif variant == "b_tilt":
        tilt = base * spec.tilt_target_db + alpha * (
            spec.tilt_target_db * (1.0 - base))
    elif variant != "anchor":
        raise ValueError(f"未知版本 {variant!r}")
    return {
        "freqs": freqs,
        "bws": np.tile(np.asarray(spec.anchor_bws, float), (t.size, 1)),
        "tilt": tilt,
        "f0": f0_track(spec, f0),
        "noise": breath_envelope(spec),
        "voiced": np.ones(t.size),
    }


def _min_harmonic_distance(f0: float, formants: np.ndarray,
                           f_max: float = MAX_ANALYSIS_HZ) -> float:
    if f0 <= 0:
        return 0.0
    ks = np.arange(1, int(f_max / f0) + 1)
    harmonics = ks * f0
    return float(min(np.min(np.abs(harmonics - fc)) for fc in formants))


def choose_safe_f0(spec: ExerciseSpec, requested: float,
                   n_samples: int = 121) -> dict:
    """在请求值附近寻找一个“稳定段谐波不压在共振峰上”的基频。

    只在滑音前后的两个稳定段评分：滑音过程中谐波必然扫过共振峰，
    那是演示的一部分，不应据此否定候选。评分取两个稳定段里
    “最近谐波距离 / f0”的较小值，越大越好；同时要求 f0 与滑音后的
    1.25×f0 都满足。平局时取更接近请求值的候选。
    """
    candidates = requested * (1.0 + np.linspace(-0.12, 0.12, n_samples))
    anchor = np.asarray(spec.anchor_freqs, float)
    moved = np.asarray(spec.moved_freqs, float)
    f_max = min(MAX_ANALYSIS_HZ, 0.45 * spec.fs)

    best = None
    for f0 in candidates:
        worst_ratio, worst_offset = np.inf, np.inf
        for ratio, band in ((1.0, anchor), (spec.f0_glide_ratio, moved)):
            pitch = f0 * ratio
            d = _min_harmonic_distance(pitch, band, f_max)
            if d < worst_offset:
                worst_offset = d
            worst_ratio = min(worst_ratio, d / pitch)
        score = (worst_ratio, -abs(f0 - requested))
        if best is None or score > best[0]:
            best = (score, f0, worst_ratio, worst_offset)
    _, f0_used, ratio, offset = best
    return {
        "f0_requested": float(requested),
        "f0_used": float(round(f0_used, 2)),
        "clearance_ratio": float(ratio),
        "worst_harmonic_offset_hz": float(offset),
        "safe": bool(ratio >= spec.clearance_target),
        "clearance_target": float(spec.clearance_target),
        "scored_segments": "pre-glide anchor and post-glide target only",
    }


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------
@dataclass
class BuiltExercise:
    spec: ExerciseSpec
    f0: float
    f0_info: dict
    times: np.ndarray
    tracks: dict
    renders: dict = field(default_factory=dict)
    answer: dict = field(default_factory=dict)

    @property
    def phrase_seconds(self) -> float:
        return self.spec.phrase_seconds


VARIANTS = ("anchor", "a_formant", "b_tilt")


def build_exercise(spec: ExerciseSpec, f0: float | None = None,
                   analysis: bool = True) -> BuiltExercise:
    """从参数完整重建练习：解析安全 f0、生成三层轨迹、渲染并分析。"""
    requested = float(spec.f0_requested if f0 is None else f0)
    info = choose_safe_f0(spec, requested)

    tracks, renders = {}, {}
    for variant in VARIANTS:
        tr = layer_tracks(spec, variant, info["f0_used"])
        params = SynthParams(
            f0=info["f0_used"], freqs=np.asarray(spec.anchor_freqs),
            bws=np.asarray(spec.anchor_bws), open_quotient=spec.open_quotient,
            shape=spec.shape, noise_gain=spec.noise_floor, tilt_db=0.0)
        # 用同一 seed 重建声源，保证三个版本的气声噪声逐样本相同
        result = render(
            spec.phrase_seconds, spec.fs, params, block_size=spec.block,
            f0_envelope=tr["f0"], freq_envelope=tr["freqs"],
            bw_envelope=tr["bws"], tilt_envelope=tr["tilt"],
            noise_envelope=tr["noise"], voiced_envelope=tr["voiced"],
            normalize=True, seed=spec.seed)
        tracks[variant], renders[variant] = tr, result

    built = BuiltExercise(spec=spec, f0=info["f0_used"], f0_info=info,
                          times=block_times(spec), tracks=tracks,
                          renders=renders)
    if analysis:
        built.answer = analyse_exercise(built)
    return built


# --------------------------------------------------------------------------
# 实际输出的分析
# --------------------------------------------------------------------------
def harmonic_amplitudes(y: np.ndarray, fs: float, t_center: float, f0: float,
                        f_max: float = MAX_ANALYSIS_HZ,
                        win_seconds: float = 0.06
                        ) -> tuple[np.ndarray, np.ndarray]:
    """在 t_center 附近用 Hann 窗 DFT 取各谐波的实测线性幅度。"""
    y = np.asarray(y, float)
    n0 = int(round(t_center * fs))
    half = max(8, int(win_seconds * fs / 2))
    i0, i1 = max(0, n0 - half), min(y.size, n0 + half)
    if i1 - i0 < 8:
        return np.array([]), np.array([])
    n = i1 - i0
    w = windows.hann(n, sym=False)
    tau = (np.arange(i0, i1) - (i0 + i1 - 1) / 2.0) / fs
    ks = np.arange(1, int(f_max / f0) + 1)
    freqs = ks * f0
    coeff = np.exp(-2j * np.pi * freqs[:, None] * tau[None, :]) @ (y[i0:i1] * w)
    amps = 2.0 * np.abs(coeff) / max(w.sum(), 1e-12)
    return freqs, amps


def envelope_db(freqs: np.ndarray, bws: np.ndarray, fs: float,
                at_hz: np.ndarray) -> np.ndarray:
    cas = FormantCascade(len(freqs), fs)
    fr, db = cas.frequency_response(freqs, bws, n_fft=8192)
    return np.interp(at_hz, fr, db)


def spectral_shape_distance(y_a: np.ndarray, y_b: np.ndarray, fs: float,
                            hop: int = 256, n_fft: int = 1024,
                            band: tuple = BAND_HZ) -> tuple[np.ndarray, np.ndarray]:
    """逐帧谱形距离（dB 域、去均值），与整体电平无关。"""
    y_a, y_b = np.asarray(y_a, float), np.asarray(y_b, float)
    n = min(y_a.size, y_b.size)
    hop = max(1, hop)
    starts = np.arange(0, max(1, n - n_fft + 1), hop)
    if starts.size == 0:
        starts = np.array([0])
    win = windows.hann(n_fft, sym=False)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    out = []
    for s in starts:
        seg_a = y_a[s:s + n_fft]
        seg_b = y_b[s:s + n_fft]
        if seg_a.size < n_fft:
            seg_a = np.pad(seg_a, (0, n_fft - seg_a.size))
            seg_b = np.pad(seg_b, (0, n_fft - seg_b.size))
        db_a = 20.0 * np.log10(np.abs(np.fft.rfft(seg_a * win)) + 1e-9)
        db_b = 20.0 * np.log10(np.abs(np.fft.rfft(seg_b * win)) + 1e-9)
        da = db_a[mask] - db_a[mask].mean()
        dbb = db_b[mask] - db_b[mask].mean()
        out.append(float(np.sqrt(np.mean((da - dbb) ** 2))))
    times = (starts + n_fft / 2.0) / fs
    return times, np.asarray(out)


def detect_divergence(times: np.ndarray, dist: np.ndarray,
                      threshold_db: float = 3.0,
                      sustain: int = 3) -> float:
    """首个持续超过阈值的帧时刻；找不到返回 nan。"""
    above = dist > threshold_db
    run = 0
    for i, flag in enumerate(above):
        run = run + 1 if flag else 0
        if run >= sustain:
            return float(times[i - sustain + 1])
    return float("nan")


def analyse_exercise(built: BuiltExercise, probes: tuple | None = None) -> dict:
    spec = built.spec
    fs, block = spec.fs, spec.block
    t = built.times
    if probes is None:
        probes = (
            spec.lead + spec.hold * 0.5,
            spec.glide_start + 0.12,
            spec.glide_start + spec.glide * 0.5,
            spec.glide_end - 0.06,
            spec.glide_end + spec.hold2 * 0.5,
            spec.phrase_seconds - 0.10,
        )
    probes = [float(max(0.02, min(spec.phrase_seconds - 0.02, p))) for p in probes]

    tracks_out = {
        "f0_hz": [round(float(v), 4) for v in built.tracks["anchor"]["f0"]],
        "noise_gain": [round(float(v), 6) for v in built.tracks["anchor"]["noise"]],
    }
    for variant in VARIANTS:
        tr = built.tracks[variant]
        tracks_out[variant] = {
            "freqs_hz": np.round(tr["freqs"], 3).tolist(),
            "bws_hz": np.round(tr["bws"], 3).tolist(),
            "tilt_db": np.round(tr["tilt"], 4).tolist(),
        }

    probe_out = {}
    divergence = {}
    ref_y = built.renders["anchor"]["y"]
    pair_dist = {}
    for variant in VARIANTS:
        frames = []
        res = built.renders[variant]
        for tp in probes:
            b = int(np.clip(round(tp * fs / block - 0.5), 0, t.size - 1))
            f0_local = float(built.tracks[variant]["f0"][b])
            fr_hz = built.tracks[variant]["freqs"][b]
            bw_hz = built.tracks[variant]["bws"][b]
            harm_f, harm_a = harmonic_amplitudes(res["y"], fs, tp, f0_local)
            harm_db = 20.0 * np.log10(harm_a + 1e-12)
            # 声道包络在各谐波处的 dB（答案派生的解释量）
            env_at_h = envelope_db(fr_hz, bw_hz, fs, harm_f)
            grid = np.linspace(100, MAX_ANALYSIS_HZ, 1600)
            peaks_idx, _ = find_peaks(
                envelope_db(fr_hz, bw_hz, fs, grid), prominence=0.5)
            peaks = grid[peaks_idx]
            nearest = [float(np.min(np.abs(harm_f - fc))) for fc in fr_hz]
            strongest = float(harm_f[int(np.argmax(harm_db))]) if harm_f.size else float("nan")
            off_to_f = [float(abs(strongest - fc)) for fc in fr_hz]
            frames.append({
                "t_s": round(tp, 4),
                "f0_hz": round(f0_local, 3),
                "envelope_peaks_hz": [round(float(v), 1) for v in peaks],
                "harmonic_freqs_hz": [round(float(v), 1) for v in harm_f],
                "harmonic_db": [round(float(v), 2) for v in harm_db],
                "envelope_at_harmonics_db": [round(float(v), 2) for v in env_at_h],
                "nearest_harmonic_offset_hz": [round(v, 1) for v in nearest],
                "strongest_harmonic_hz": round(strongest, 1),
                "strongest_to_formant_offset_hz": [round(v, 1) for v in off_to_f],
            })
        probe_out[variant] = frames

        dist_t, dist = spectral_shape_distance(
            res["y"], ref_y, fs, hop=block, band=BAND_HZ)
        divergence[variant] = {
            "onset_s": detect_divergence(dist_t, dist),
            "threshold_db": 3.0,
            "scheduled_glide_start_s": round(spec.glide_start, 4),
            "distance_db": [round(float(v), 3) for v in dist],
            "distance_times_s": [round(float(v), 4) for v in dist_t],
            "pre_glide_median_db": round(float(np.median(
                dist[dist_t < spec.glide_start])), 4) if np.any(
                    dist_t < spec.glide_start) else None,
            "post_glide_median_db": round(float(np.median(
                dist[dist_t > spec.glide_end])), 4) if np.any(
                    dist_t > spec.glide_end) else None,
        }

    # 关键比较：A 组与 B 组直接对比（滑音前应接近，滑音后应拉开）
    pair_t, pair_d = spectral_shape_distance(
        built.renders["a_formant"]["y"], built.renders["b_tilt"]["y"],
        fs, hop=block, band=BAND_HZ)
    pair_dist = {
        "onset_s": detect_divergence(pair_t, pair_d),
        "threshold_db": 3.0,
        "distance_db": [round(float(v), 3) for v in pair_d],
        "distance_times_s": [round(float(v), 4) for v in pair_t],
        "pre_glide_median_db": round(float(np.median(
            pair_d[pair_t < spec.glide_start])), 4) if np.any(
                pair_t < spec.glide_start) else None,
        "glide_median_db": round(float(np.median(
            pair_d[(pair_t >= spec.glide_start) & (pair_t <= spec.glide_end)])), 4)
            if np.any((pair_t >= spec.glide_start) & (pair_t <= spec.glide_end))
            else None,
        "post_glide_median_db": round(float(np.median(
            pair_d[pair_t > spec.glide_end])), 4) if np.any(
                pair_t > spec.glide_end) else None,
    }

    return {
        "name": spec.name,
        "fs": fs,
        "block": block,
        "f0": {
            "requested_hz": built.f0_info["f0_requested"],
            "used_hz": built.f0_info["f0_used"],
            "glide_ratio": spec.f0_glide_ratio,
            "clearance_ratio": round(built.f0_info["clearance_ratio"], 4),
            "worst_harmonic_offset_hz": round(
                built.f0_info["worst_harmonic_offset_hz"], 2),
            "safe": built.f0_info["safe"],
        },
        "timeline": {
            "lead_s": spec.lead, "hold_s": spec.hold,
            "glide_start_s": round(spec.glide_start, 4),
            "glide_end_s": round(spec.glide_end, 4),
            "hold2_s": spec.hold2,
            "phrase_s": round(spec.phrase_seconds, 4),
        },
        "layers_changed": {
            "a_formant": "tract.formant_centers",
            "b_tilt": "source.spectral_tilt",
            "anchor": "none (shared reference)",
        },
        "baseline_frac": spec.baseline_frac,
        "tracks": tracks_out,
        "breath_events": [{
            "start_s": round(max(0.0, spec.glide_start + spec.breath_delay
                                 - spec.breath_width), 4),
            "end_s": round(spec.glide_start + spec.breath_delay
                           + spec.breath_width, 4),
            "peak_gain": spec.breath_peak,
            "note": "raised-cosine breath event; identical sample-for-sample "
                    "across all three variants (same f0 and noise seed)",
        }],
        "probes": probe_out,
        "divergence": divergence,
        "pair_a_vs_b": pair_dist,
    }


# --------------------------------------------------------------------------
# 跨基频核验：同一份声道答案在另一 f0 下是否仍能解释实际输出
# --------------------------------------------------------------------------
def _explainability(res: dict, spec: ExerciseSpec, variant: str,
                    probes: list[float]) -> dict:
    """比较“实测输出谐波”与“源谐波 + 声道包络”的预测。"""
    y, src = res["y"], res["source"]
    fs = spec.fs
    rows = []
    for tp in probes:
        b = int(np.clip(round(tp * fs / spec.block - 0.5), 0,
                        len(res["freqs"]) - 1))
        f0_local = float(res["f0_track"][min(int(tp * fs),
                                             res["f0_track"].size - 1)])
        fr_hz, bw_hz = res["freqs"][b], res["bws"][b]
        hf, ha = harmonic_amplitudes(y, fs, tp, f0_local)
        _, sa = harmonic_amplitudes(src, fs, tp, f0_local)
        if hf.size == 0:
            continue
        actual = 20.0 * np.log10(ha + 1e-12)
        source_db = 20.0 * np.log10(sa + 1e-12)
        pred = source_db + envelope_db(fr_hz, bw_hz, fs, hf)
        a = actual - actual.mean()
        p = pred - pred.mean()
        corr = float(np.corrcoef(a, p)[0, 1]) if a.size > 2 else float("nan")
        rmse = float(np.sqrt(np.mean((a - p) ** 2)))
        rows.append({"t_s": round(float(tp), 4), "corr": round(corr, 4),
                     "rmse_db": round(rmse, 3),
                     "n_harmonics": int(hf.size)})
    return {"variant": variant, "frames": rows}


def cross_pitch_check(built: BuiltExercise, alt_f0: float,
                      tolerance_db: float = 3.0) -> dict:
    """用另一基频重新生成实际输出，核对答案声道包络是否仍可解释。

    做法：同一声道轨迹在 alt_f0 下重新渲染；对滑音后的若干帧，
    预测“输出谐波 = 源谱(含倾斜) + 声道包络”，与实际输出谐波比较。
    同时给出“直接取最强谐波当共振峰”的陷阱：它在不同 f0 下会跳变，
    而答案里的包络峰位不随 f0 改变。
    """
    spec = built.spec
    alt = build_exercise(spec, f0=alt_f0, analysis=False)
    probes = [spec.glide_end + 0.08,
              spec.glide_end + (spec.phrase_seconds - spec.glide_end) * 0.5,
              spec.phrase_seconds - 0.10]
    rows = {v: _explainability(alt.renders[v], spec, v, probes)
            for v in VARIANTS}

    # 包络峰位（来自参数答案）在任何 f0 下都相同
    b_post = len(alt.times) - 3
    peaks = {}
    for v in ("anchor", "a_formant"):
        fr = alt.tracks[v]["freqs"][b_post]
        bw = alt.tracks[v]["bws"][b_post]
        grid = np.linspace(100, MAX_ANALYSIS_HZ, 1600)
        db = envelope_db(fr, bw, spec.fs, grid)
        idx, _ = find_peaks(db, prominence=0.5)
        peaks[v] = [round(float(x), 1) for x in grid[idx]]

    # 朴素“最强谐波”估计在不同 f0 下会移动；包络答案不会
    naive = {}
    for v in ("anchor", "a_formant"):
        fr = alt.tracks[v]["freqs"][b_post]
        hf, ha = harmonic_amplitudes(alt.renders[v]["y"], spec.fs,
                                     spec.phrase_seconds - 0.10,
                                     float(alt.tracks[v]["f0"][b_post]))
        pick = []
        for fc in fr:
            if hf.size:
                j = int(np.argmin(np.abs(hf - fc)))
                pick.append(round(float(hf[j]), 1))
            else:
                pick.append(float("nan"))
        naive[v] = pick

    ok = all(np.isfinite([r["corr"] for r in rows[v]["frames"]]).all()
             and all(r["corr"] > 0.9 and r["rmse_db"] < tolerance_db
                     for r in rows[v]["frames"])
             for v in ("anchor", "a_formant"))
    return {
        "alt_f0_requested_hz": float(alt_f0),
        "alt_f0_used_hz": alt.f0,
        "alt_clearance_ratio": round(alt.f0_info["clearance_ratio"], 4),
        "probes_s": [round(p, 4) for p in probes],
        "explainability": rows,
        "envelope_peaks_hz": peaks,
        "nearest_harmonic_to_formant_hz": naive,
        "tolerance_db": tolerance_db,
        "ok": bool(ok),
        "note": ("包络峰位由声道参数决定，不随 f0 改变；"
                 "最靠近共振峰的谐波会随 f0 移动，不能当作共振峰。"),
    }


# --------------------------------------------------------------------------
# 音高穿过两个共振带之间的示范
# --------------------------------------------------------------------------
@dataclass
class PitchDemoSpec:
    fs: int = FS_DEFAULT
    duration: float = 2.6
    block: int = BLOCK_DEFAULT
    f0_start: float = 380.0
    f0_end: float = 1100.0
    freqs: tuple = (520.0, 980.0, 2600.0, 3400.0)
    bws: tuple = (55.0, 70.0, 140.0, 180.0)
    open_quotient: float = 0.62
    noise_gain: float = 0.0
    tilt_db: float = 0.0


def build_pitch_demo(spec: PitchDemoSpec) -> dict:
    n = int(round(spec.duration * spec.fs))
    n_blocks = int(np.ceil(n / spec.block))
    f0_env = np.linspace(spec.f0_start, spec.f0_end, n_blocks)
    params = SynthParams(f0=spec.f0_start,
                         freqs=np.asarray(spec.freqs),
                         bws=np.asarray(spec.bws),
                         open_quotient=spec.open_quotient,
                         noise_gain=spec.noise_gain, tilt_db=spec.tilt_db)
    res = render(spec.duration, spec.fs, params, block_size=spec.block,
                 f0_envelope=f0_env)

    # 逐帧：谐波间距、最强谐波、真实包络峰
    frames = []
    step = 0.05
    times = np.arange(0.10, spec.duration - 0.10, step)
    grid = np.linspace(100, MAX_ANALYSIS_HZ, 1600)
    env = envelope_db(np.asarray(spec.freqs), np.asarray(spec.bws),
                      spec.fs, grid)
    pk, _ = find_peaks(env, prominence=0.5)
    env_peaks = grid[pk]
    for tp in times:
        n0 = int(tp * spec.fs)
        f0_local = float(res["f0_track"][min(n0, res["f0_track"].size - 1)])
        hf, ha = harmonic_amplitudes(res["y"], spec.fs, tp, f0_local,
                                     win_seconds=0.05)
        if hf.size == 0:
            continue
        strongest = float(hf[int(np.argmax(ha))])
        frames.append({
            "t_s": round(float(tp), 3),
            "f0_hz": round(f0_local, 2),
            "strongest_harmonic_hz": round(strongest, 2),
            "strongest_harmonic_db": round(
                float(20 * np.log10(ha.max() + 1e-12)), 2),
            "between_bands": bool(spec.freqs[0] < f0_local < spec.freqs[1]),
            "strongest_offset_to_f1_hz": round(abs(strongest - spec.freqs[0]), 1),
            "strongest_offset_to_f2_hz": round(abs(strongest - spec.freqs[1]), 1),
        })

    # 挑一个 f0 落在两个共振带之间、基波就是最强谐波，且离两个中心都最远的帧
    candidates = [f for f in frames
                  if f["between_bands"]
                  and abs(f["strongest_harmonic_hz"] - f["f0_hz"]) < 0.6]
    focus = max(candidates,
                key=lambda f: min(f["strongest_offset_to_f1_hz"],
                                  f["strongest_offset_to_f2_hz"])) \
        if candidates else None

    return {
        "spec": asdict(spec),
        "env_freqs_hz": np.round(grid, 1).tolist(),
        "env_db": np.round(env, 2).tolist(),
        "envelope_peaks_hz": [round(float(v), 1) for v in env_peaks],
        "frames": frames,
        "focus_frame": focus,
        "finding": (
            "f0 sweeps {:.0f} -> {:.0f} Hz and passes F1={:.0f} Hz and "
            "F2={:.0f} Hz in turn. The tract envelope peaks stay at "
            "{:.0f}/{:.0f} Hz, yet at f0={:.1f} Hz the fundamental sits "
            "between the two bands and is the strongest harmonic. A local "
            "maximum in the harmonic spectrum is therefore not the center "
            "of a vocal-tract resonance.").format(
                spec.f0_start, spec.f0_end, spec.freqs[0], spec.freqs[1],
                env_peaks[0] if env_peaks.size else float("nan"),
                env_peaks[1] if env_peaks.size > 1 else float("nan"),
                focus["f0_hz"] if focus else float("nan")),
        "render": res,
    }


# --------------------------------------------------------------------------
# 资产写出
# --------------------------------------------------------------------------
def _concat_pair(a: np.ndarray, b: np.ndarray, fs: int,
                 gap: float = 0.35) -> np.ndarray:
    silence = np.zeros(int(round(gap * fs)))
    return np.concatenate([a, silence, b])


def save_assets(built: BuiltExercise, out_dir: str, *,
                hide_answer: bool = False, plots: bool = True,
                pair_gap: float = 0.35) -> dict:
    """写出 WAV、参数与（可选的）答案轨迹。返回文件清单。"""
    os.makedirs(out_dir, exist_ok=True)
    fs = built.spec.fs
    files = {}
    for variant in VARIANTS:
        path = os.path.join(out_dir, f"{variant}.wav")
        write_wav(path, built.renders[variant]["y"], fs)
        files[variant] = path

    for name, target in (("pairA_anchor_then_a_formant", "a_formant"),
                         ("pairB_anchor_then_b_tilt", "b_tilt")):
        path = os.path.join(out_dir, f"{name}.wav")
        write_wav(path, _concat_pair(built.renders["anchor"]["y"],
                                     built.renders[target]["y"], fs,
                                     pair_gap), fs)
        files[name] = path

    spec_path = os.path.join(out_dir, "spec.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump({
            "spec": built.spec.to_dict(),
            "resolved_f0_hz": built.f0,
            "f0_clearance": built.f0_info,
            "answer_hidden": bool(hide_answer),
        }, f, ensure_ascii=False, indent=2)
    files["spec"] = spec_path

    if not hide_answer:
        ans_path = os.path.join(out_dir, "answer.json")
        with open(ans_path, "w", encoding="utf-8") as f:
            json.dump(built.answer, f, ensure_ascii=False)
        files["answer"] = ans_path

    if plots:
        fig_path = os.path.join(out_dir, "exercise_overview.png")
        plot_exercise(built, fig_path, show_answer=not hide_answer)
        files["plot"] = fig_path

    return files


def save_pitch_demo(demo: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    spec = PitchDemoSpec(**demo["spec"])
    wav = os.path.join(out_dir, "pitch_between_formants.wav")
    write_wav(wav, demo["render"]["y"], spec.fs)
    meta = {k: demo[k] for k in ("envelope_peaks_hz", "focus_frame",
                                 "finding", "spec")}
    meta["frames"] = demo["frames"]
    js = os.path.join(out_dir, "pitch_between_formants.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    png = os.path.join(out_dir, "pitch_between_formants.png")
    plot_pitch_demo(demo, png)
    return {"wav": wav, "json": js, "plot": png}


# --------------------------------------------------------------------------
# 图
# --------------------------------------------------------------------------
def _spectrogram(y: np.ndarray, fs: float, n_fft: int = 1024,
                 hop: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(0, max(1, len(y) - n_fft + 1), hop)
    win = windows.hann(n_fft, sym=False)
    S = np.array([np.abs(np.fft.rfft(y[s:s + n_fft] * win))
                  for s in starts]).T
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    times = (starts + n_fft / 2) / fs
    return freqs, times, 20.0 * np.log10(S + 1e-9)


def plot_exercise(built: BuiltExercise, path: str, *,
                  show_answer: bool = True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = built.spec
    fs = spec.fs
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.2]})
    variants = [("anchor", "anchor (shared)"),
                ("a_formant", "A: only formants move"),
                ("b_tilt", "B: only source tilt changes")]
    for ax, (v, title) in zip(axes[:3], variants):
        fr, tt, S = _spectrogram(built.renders[v]["y"], fs)
        band = fr <= 4200
        sub = S[band]
        # 每个版本单独定标：倾斜把高频压暗时仍能看清谐波走向
        vmax = float(np.percentile(sub, 99.9))
        vmin = float(np.percentile(sub, 35))
        ax.pcolormesh(tt, fr[band], sub, shading="auto", cmap="magma",
                      vmin=vmin, vmax=vmax)
        ax.set_ylabel("Hz")
        ax.set_title(title, fontsize=10, loc="left")
        if show_answer:
            for k in range(4):
                ax.plot(built.times, built.tracks[v]["freqs"][:, k],
                        color="cyan", lw=1.0, ls="--", alpha=0.9,
                        label=f"F{k + 1}" if v == "anchor" else None)
            if v == "b_tilt":
                ax2 = ax.twinx()
                ax2.plot(built.times, built.tracks[v]["tilt"], color="lime",
                         lw=1.2)
                ax2.set_ylabel("tilt dB", color="lime")
                ax2.tick_params(axis="y", colors="lime")
    if show_answer:
        axes[0].legend(loc="upper right", fontsize=7, ncol=4)

    ax = axes[3]
    for v, label, c in (("a_formant", "A vs anchor", "tab:orange"),
                        ("b_tilt", "B vs anchor", "tab:blue")):
        div = built.answer["divergence"][v]
        ax.plot(div["distance_times_s"], div["distance_db"], lw=1.2,
                label=label, color=c, alpha=0.65)
    pair = built.answer["pair_a_vs_b"]
    ax.plot(pair["distance_times_s"], pair["distance_db"], lw=2.0,
            color="k", label="A vs B")
    for v, c in (("a_formant", "tab:orange"), ("b_tilt", "tab:blue"),
                 (None, "k")):
        div = pair if v is None else built.answer["divergence"][v]
        onset = div["onset_s"]
        if np.isfinite(onset):
            ax.axvline(onset, color=c, ls=":", lw=1.0)
    for tline, ls in ((spec.glide_start, "--"), (spec.glide_end, "--")):
        ax.axvline(tline, color="k", ls=ls, lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("spectral shape distance (dB)")
    ax.set_title("Measured from the rendered audio: A vs anchor, B vs anchor, "
                 "and A vs B (dashed = scheduled glide)", fontsize=10,
                 loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_pitch_demo(demo: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = PitchDemoSpec(**demo["spec"])
    res = demo["render"]
    fs = spec.fs
    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                             gridspec_kw={"height_ratios": [1.5, 1]})
    fr, tt, S = _spectrogram(res["y"], fs, n_fft=2048, hop=128)
    band = fr <= 1500
    ax = axes[0]
    sb = S[band]
    floor = float(np.percentile(sb, 40))
    ax.pcolormesh(tt, fr[band], S[band], shading="auto", cmap="magma",
                  vmin=floor, vmax=float(np.percentile(sb, 99.9)))
    for f in spec.freqs[:2]:
        ax.axhline(f, color="cyan", ls="--", lw=1.2)
    focus = demo.get("focus_frame")
    if focus:
        ax.axvline(focus["t_s"], color="white", ls=":", lw=1.0)
        ax.plot(focus["t_s"], focus["strongest_harmonic_hz"], "o",
                mfc="none", mec="lime", ms=9)
        ax.annotate(
            f"f0={focus['f0_hz']:.0f} Hz is between F1/F2\n"
            f"and is the strongest harmonic",
            (focus["t_s"], focus["strongest_harmonic_hz"]),
            xytext=(focus["t_s"] + 0.15, 1250), color="white", fontsize=8,
            arrowprops=dict(color="white", arrowstyle="->", lw=0.8))
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Pitch glides through the gap between two formant bands "
                 "(cyan = F1/F2)", fontsize=11)

    ax = axes[1]
    grid = np.array(demo["env_freqs_hz"])
    env = np.array(demo["env_db"])
    ax.plot(grid, env, color="tab:purple", lw=1.4, label="vocal-tract envelope")
    ax.axvline(spec.freqs[0], color="cyan", ls="--", lw=1.0, label="F1")
    ax.axvline(spec.freqs[1], color="cyan", ls="--", lw=1.0, label="F2")
    if focus:
        ax.plot(focus["strongest_harmonic_hz"], focus["strongest_harmonic_db"],
                "o", mfc="none", mec="tab:red", ms=9,
                label="strongest harmonic (focus frame)")
    ax.set_xlim(0, 1500)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("dB")
    ax.legend(fontsize=8)
    ax.set_title(demo["finding"], fontsize=9, loc="left", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
