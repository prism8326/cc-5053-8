"""乐器窗口：暂停、固定某一层再演奏，由实际输出显示差异从何处开始。

三个可固定的层共享同一条 f0 滑音与同一次气声事件：

- ``source.tilt``   声源层：源谱倾斜（B 组的变化层）
- ``tract.formants`` 声道层：共振峰移动（A 组的变化层）
- ``air.breath``    气声事件层

每个层可以勾选 **Fix**：固定表示冻结在滑音前的参照取值。窗口对每个版本
并行渲染 “当前设置” 与 “三层全部冻结的静态参照”，并比较两者的实际波形
谱形。于是：

- 只固定 ``tract.formants`` 后，A 的差异曲线归零 → A 的差异全部来自声道层；
- 只固定 ``source.tilt`` 后，B 的差异曲线归零 → B 的差异全部来自声源层；
- 不固定时，两条曲线都在同一个时刻抬起，说明差异从滑音段开始。

所有曲线都从当前参数即时渲染与测量，不读取任何预录答案；“Hide answer”
只隐藏参数轨迹，实测谱形与差异曲线仍在，便于创作者盲听核对。
"""

from __future__ import annotations

import os

import matplotlib

if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider  # noqa: E402

from .audio_io import BufferPlayer, sounddevice_available  # noqa: E402
from .engine import SynthParams, render  # noqa: E402
from .exercise import (BAND_HZ, ExerciseSpec, build_exercise,  # noqa: E402
                       detect_divergence, f0_track, layer_tracks,
                       spectral_shape_distance)

LAYERS = ("source.tilt", "tract.formants", "air.breath")
MONITOR_CHOICES = ("A", "B", "A+B pair")


class InstrumentWindow:
    def __init__(self, spec: ExerciseSpec | None = None,
                 f0: float | None = None):
        self.spec = spec or ExerciseSpec()
        if f0 is not None:
            self.spec.f0_requested = float(f0)
        base = build_exercise(self.spec, analysis=False)
        self.f0 = base.f0
        self.fixed = {name: False for name in LAYERS}
        self.show_answer = True
        self.monitor = "A"
        self.cursor = self.spec.glide_start
        self.player = BufferPlayer(np.zeros(16), self.spec.fs)
        self._build_figure()
        self._render_and_analyse()
        self._start_timer()

    # ---------------- layout ----------------
    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle(
            "Instrument window: pause, fix one layer, play again — "
            "the actual output shows where the difference begins",
            fontsize=13)
        gs = self.fig.add_gridspec(2, 2, left=0.07, right=0.97, top=0.92,
                                   bottom=0.44, hspace=0.42, wspace=0.24)
        self.ax_spec = self.fig.add_subplot(gs[0, 0])
        self.ax_traj = self.fig.add_subplot(gs[0, 1])
        self.ax_dist = self.fig.add_subplot(gs[1, 0])
        self.ax_state = self.fig.add_subplot(gs[1, 1])

        self.ax_layers = self.fig.add_axes([0.07, 0.235, 0.18, 0.11])
        self.ax_monitor = self.fig.add_axes([0.27, 0.235, 0.10, 0.11])
        self.ax_time = self.fig.add_axes([0.47, 0.285, 0.48, 0.022])
        self.slider_time = Slider(self.ax_time, "scrub (s)", 0.0,
                                  self.spec.phrase_seconds,
                                  valinit=self.cursor)
        self.slider_time.on_changed(self._on_scrub)

        self.btn_play = Button(self.fig.add_axes([0.07, 0.10, 0.13, 0.05]),
                               "Play / Pause")
        self.btn_play.on_clicked(self._toggle_play)
        self.btn_render = Button(
            self.fig.add_axes([0.22, 0.10, 0.18, 0.05]),
            "Render & analyse")
        self.btn_render.on_clicked(lambda _e: self._render_and_analyse())
        self.btn_reset = Button(
            self.fig.add_axes([0.42, 0.10, 0.12, 0.05]), "Reset layers")
        self.btn_reset.on_clicked(self._reset_layers)
        self.btn_answer = Button(
            self.fig.add_axes([0.56, 0.10, 0.15, 0.05]), "Hide answer")
        self.btn_answer.on_clicked(self._toggle_answer)

        self.status = self.fig.add_axes([0.07, 0.015, 0.88, 0.06])
        self.status.axis("off")
        self.status_txt = self.status.text(0.0, 1.0, "", va="top",
                                          fontsize=8.5, family="monospace")
        self._make_layer_checks()
        self.monitor_radio = RadioButtons(self.ax_monitor, MONITOR_CHOICES,
                                          active=0)
        self.monitor_radio.on_clicked(self._on_monitor)
        self.ax_monitor.set_title("listen to", fontsize=8)
        for lbl in self.monitor_radio.labels:
            lbl.set_fontsize(8)

    def _make_layer_checks(self) -> None:
        self.ax_layers.clear()
        self.layers_check = CheckButtons(
            self.ax_layers, [f"Fix {n}" for n in LAYERS],
            [self.fixed[n] for n in LAYERS])
        self.layers_check.on_clicked(self._on_layer_toggle)
        self.ax_layers.set_title("freeze a layer at its pre-glide value",
                                 fontsize=8)
        for lbl in self.layers_check.labels:
            lbl.set_fontsize(8)

    # ---------------- controls ----------------
    def _on_layer_toggle(self, label: str) -> None:
        name = label.replace("Fix ", "")
        self.fixed[name] = not self.fixed[name]
        self._render_and_analyse()

    def _reset_layers(self, _event=None) -> None:
        self.fixed = {name: False for name in LAYERS}
        self._make_layer_checks()
        self._render_and_analyse()

    def _toggle_answer(self, _event=None) -> None:
        self.show_answer = not self.show_answer
        self.btn_answer.label.set_text(
            "Hide answer" if self.show_answer else "Show answer")
        self._redraw()

    def _on_monitor(self, label: str) -> None:
        self.monitor = label
        self._load_monitor_audio()

    def _on_scrub(self, value: float) -> None:
        self.cursor = float(value)
        if self.player.playing:
            self.player.pos = int(self.cursor * self.spec.fs)
        self._redraw()

    def _toggle_play(self, _event=None) -> None:
        if self.player.playing:
            self.player.pause()
        else:
            self.player.pos = int(self.cursor * self.spec.fs)
            self.player.play()
        self._redraw()

    def _start_timer(self) -> None:
        self.timer = self.fig.canvas.new_timer(interval=40)
        self.timer.add_callback(self._tick)
        self.timer.start()

    def _tick(self) -> None:
        if not self.player.playing:
            return
        self.player.advance(0.04)
        self.cursor = min(self.player.time, self.spec.phrase_seconds)
        self.slider_time.set_val(self.cursor)
        self._redraw()

    # ---------------- rendering ----------------
    def _tracks_with_fixes(self, variant: str) -> dict:
        tr = layer_tracks(self.spec, variant, self.f0)
        out = {k: np.array(v, copy=True) for k, v in tr.items()}
        anchor = np.asarray(self.spec.anchor_freqs, float)
        pre_glide_a = anchor + self.spec.baseline_frac * (
            np.asarray(self.spec.moved_freqs, float) - anchor)
        if self.fixed["source.tilt"]:
            out["tilt"][:] = 0.0
        if self.fixed["tract.formants"]:
            out["freqs"][:] = pre_glide_a if variant == "a_formant" else anchor
        if self.fixed["air.breath"]:
            out["noise"][:] = self.spec.noise_floor
        return out

    def _render_tracks(self, tr: dict) -> dict:
        params = SynthParams(
            f0=self.f0, freqs=np.asarray(self.spec.anchor_freqs),
            bws=np.asarray(self.spec.anchor_bws),
            open_quotient=self.spec.open_quotient,
            shape=self.spec.shape, noise_gain=self.spec.noise_floor)
        return render(self.spec.phrase_seconds, self.spec.fs, params,
                      block_size=self.spec.block, f0_envelope=tr["f0"],
                      freq_envelope=tr["freqs"], bw_envelope=tr["bws"],
                      tilt_envelope=tr["tilt"], noise_envelope=tr["noise"],
                      voiced_envelope=tr["voiced"], normalize=True,
                      seed=self.spec.seed)

    def _reference_tracks(self, variant: str, current: dict) -> dict:
        """该乐器的惰性参照：把“自己的变化层”冻结在滑音前，其余完全相同。

        这样 A 的曲线只反映声道层的变化、B 的曲线只反映源谱倾斜的变化；
        共享的 f0 滑音与气声事件在两侧同时出现而抵消。
        """
        out = {k: np.array(v, copy=True) for k, v in current.items()}
        if variant == "a_formant":
            out["freqs"][:] = current["freqs"][0]
        else:
            out["tilt"][:] = current["tilt"][0]
        return out

    def _render_and_analyse(self) -> None:
        self.current, self.reference = {}, {}
        tracks_current = {}
        for variant in ("a_formant", "b_tilt"):
            tr = self._tracks_with_fixes(variant)
            tracks_current[variant] = tr
            self.current[variant] = self._render_tracks(tr)
            ref_tr = self._reference_tracks(variant, tr)
            self.reference[variant] = self._render_tracks(ref_tr)
        fs, hop = self.spec.fs, self.spec.block
        self.dist = {}
        for variant in ("a_formant", "b_tilt"):
            t, d = spectral_shape_distance(self.current[variant]["y"],
                                           self.reference[variant]["y"], fs,
                                           hop=hop, band=BAND_HZ)
            self.dist[variant] = {
                "t": t, "d": d, "onset": detect_divergence(t, d)}
        t, d = spectral_shape_distance(self.current["a_formant"]["y"],
                                       self.current["b_tilt"]["y"], fs,
                                       hop=hop, band=BAND_HZ)
        self.dist["a_vs_b"] = {"t": t, "d": d,
                               "onset": detect_divergence(t, d)}
        self._load_monitor_audio()

    def _load_monitor_audio(self) -> None:
        a = self.current["a_formant"]["y"]
        b = self.current["b_tilt"]["y"]
        if self.monitor == "B":
            audio = b
        elif self.monitor == "A+B pair":
            gap = np.zeros(int(0.3 * self.spec.fs))
            audio = np.concatenate([a, gap, b])
        else:
            audio = a
        was_playing = self.player.playing
        self.player.load(audio)
        if was_playing:
            self.player.pos = int(self.cursor * self.spec.fs)
            self.player.play()
        self._redraw()

    # ---------------- plotting ----------------
    def _spec_times(self) -> np.ndarray:
        n = int(round(self.spec.phrase_seconds * self.spec.fs
                      / self.spec.block))
        return (np.arange(n) + 0.5) * self.spec.block / self.spec.fs

    def _block_index(self) -> int:
        n = int(round(self.cursor * self.spec.fs / self.spec.block - 0.5))
        return int(np.clip(n, 0, self._spec_times().size - 1))

    def _redraw(self) -> None:
        spec = self.spec
        fs = spec.fs
        t = self._spec_times()
        bi = self._block_index()

        self.ax_spec.clear()
        seg = int(np.clip(self.cursor, 0.0, spec.phrase_seconds) * fs)
        half = int(0.06 * fs)
        n_fft = 2048
        for key, var, c, label in (("A", "a_formant", "tab:orange", "A actual"),
                                   ("B", "b_tilt", "tab:blue", "B actual")):
            y = self.current[var]["y"]
            h0, h1 = max(0, seg - half), min(y.size, seg + half)
            w = np.hanning(max(2, h1 - h0))
            mag = np.abs(np.fft.rfft(y[h0:h1] * w, n=n_fft))
            fr = np.fft.rfftfreq(n_fft, 1.0 / fs)
            band = fr <= 4500
            self.ax_spec.plot(fr[band], 20 * np.log10(mag[band] + 1e-9),
                              color=c, lw=1.0, label=label)
        self.ax_spec.set_title(f"Actual output spectra at t = "
                               f"{self.cursor:.2f} s", fontsize=10)
        self.ax_spec.set_xlabel("frequency (Hz)")
        self.ax_spec.set_ylabel("dB")
        self.ax_spec.legend(fontsize=8)

        self.ax_traj.clear()
        self.ax_traj.plot(t, f0_track(spec, self.f0), color="0.35", lw=1.0,
                          label="f0")
        if self.show_answer:
            a_tr = self._tracks_with_fixes("a_formant")
            b_tr = self._tracks_with_fixes("b_tilt")
            for k in range(4):
                self.ax_traj.plot(t, a_tr["freqs"][:, k], color="tab:orange",
                                  lw=0.9, alpha=0.75,
                                  label="A formants" if k == 0 else None)
                self.ax_traj.plot(t, b_tr["freqs"][:, k], color="tab:blue",
                                  lw=0.9, alpha=0.5,
                                  label="B formants" if k == 0 else None)
            ax2 = self.ax_traj.twinx()
            ax2.plot(t, a_tr["tilt"], color="tab:orange", ls="--", lw=1.0,
                     label="A tilt")
            ax2.plot(t, b_tr["tilt"], color="tab:blue", ls="--", lw=1.0,
                     label="B tilt")
            ax2.set_ylabel("tilt dB")
            self.ax_traj.set_title("Trajectories under current fixes",
                                   fontsize=10)
            self.ax_traj.legend(fontsize=7, loc="upper left")
        else:
            self.ax_traj.set_title("Answer hidden (spectra and distances "
                                   "still measured live)", fontsize=10)
        self.ax_traj.axvline(self.cursor, color="k", ls=":", lw=0.8)
        self.ax_traj.set_xlabel("time (s)")
        self.ax_traj.set_ylabel("Hz")

        self.ax_dist.clear()
        for key, var, c, label in (
                ("A", "a_formant", "tab:orange", "A vs its frozen reference"),
                ("B", "b_tilt", "tab:blue", "B vs its frozen reference")):
            info = self.dist[var]
            self.ax_dist.plot(info["t"], info["d"], color=c, lw=1.5,
                              label=label)
        info = self.dist["a_vs_b"]
        self.ax_dist.plot(info["t"], info["d"], color="k", lw=1.2,
                          alpha=0.55, label="A vs B")
        for var, c in (("a_formant", "tab:orange"), ("b_tilt", "tab:blue")):
            onset = self.dist[var]["onset"]
            if np.isfinite(onset):
                self.ax_dist.axvline(onset, color=c, ls="--", lw=1.0)
        for tl in (spec.glide_start, spec.glide_end):
            self.ax_dist.axvline(tl, color="0.6", ls=":", lw=0.8)
        self.ax_dist.axvline(self.cursor, color="k", ls=":", lw=0.8)
        self.ax_dist.set_xlabel("time (s)")
        self.ax_dist.set_ylabel("spectral shape distance (dB)")
        self.ax_dist.set_title("Measured from the actual output: when each "
                               "instrument departs from its frozen reference",
                               fontsize=9)
        self.ax_dist.legend(fontsize=8)

        self.ax_state.clear()
        info = self.dist["a_vs_b"]
        self.ax_state.plot(info["t"], info["d"], color="0.55", lw=1.0,
                           label="A vs B")
        self.ax_state.axvline(self.cursor, color="tab:green", lw=1.4)
        for var, c in (("a_formant", "tab:orange"), ("b_tilt", "tab:blue")):
            onset = self.dist[var]["onset"]
            if np.isfinite(onset):
                self.ax_state.axvline(onset, color=c, ls="--", lw=0.9)
        self.ax_state.set_xlabel("time (s)")
        self.ax_state.set_ylabel("A vs B (dB)")
        f0_now = float(f0_track(spec, self.f0)[bi])
        self.ax_state.set_title(
            f"cursor {self.cursor:.2f}s | f0 {f0_now:.1f}Hz | "
            f"playing={self.player.playing} | "
            f"audio {'on' if sounddevice_available() else 'offline cursor'}",
            fontsize=8)

        fixed = [n for n, v in self.fixed.items() if v]
        onsets = ", ".join(
            f"{'A' if v == 'a_formant' else 'B'}="
            f"{self.dist[v]['onset']:.2f}s"
            if np.isfinite(self.dist[v]["onset"]) else
            f"{'A' if v == 'a_formant' else 'B'}=stable"
            for v in ("a_formant", "b_tilt"))
        self.status_txt.set_text(
            "fixed layers: " + (", ".join(fixed) if fixed else "none")
            + f"  |  measured onset from frozen reference: {onsets}"
            + "  |  answer " + ("shown" if self.show_answer else "hidden")
            + ("" if sounddevice_available() else
               "  |  no sound card: cursor advances silently; "
               "export WAV to listen"))
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def show(self) -> None:
        plt.show()


class ExerciseGUI(InstrumentWindow):
    """CLI 入口别名。"""
